"""Streamlit operations dashboard for the SAC smart-charging server.

Runs as its own process next to :mod:`serve` and talks to it over HTTP:
every number on the page comes from ``GET /api/dashboard``, so the
dashboard can never drift from the schedule the charger is actually
given. Point it at another host with the ``SERVE_URL`` environment
variable.

Access is gated by Keycloak (the ``AI-Charge-Technologies`` realm, the
same one the operator tools use) via Streamlit's native OIDC support.
The realm credentials are written to ``.streamlit/secrets.toml`` at
startup by ``scripts/dashboard.sh``, which reads them from Infisical —
they are never committed. Calls to ``/api/dashboard`` additionally carry
``DASHBOARD_API_TOKEN`` as a bearer token.

Run it with::

    streamlit run dashboard_app.py --server.port 8501

or via ``scripts/dashboard.sh start``.
"""

import base64
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import altair as alt
import pandas as pd
import requests
import streamlit as st

import config
import dashboard

config.load()

SERVE_URL = os.getenv("SERVE_URL", "http://127.0.0.1:8000").rstrip("/")
ASSETS_DIR = Path(__file__).resolve().parent / "assets"
PAGE_ICON = ASSETS_DIR / "favicon-32.png"
SIGN_IN_MARK = ASSETS_DIR / "favicon-180.png"
API_TOKEN = os.getenv("DASHBOARD_API_TOKEN", "")
# Escape hatch for local development only; the deployed service leaves it unset.
ALLOW_ANONYMOUS = os.getenv("DASHBOARD_ALLOW_ANONYMOUS", "") == "1"
AUTH_PROVIDER = os.getenv("DASHBOARD_AUTH_PROVIDER", "keycloak")
REQUEST_TIMEOUT_S = 30

# ── Palette ───────────────────────────────────────────────────────────────────
# Validated for colour-vision deficiency against both chart surfaces
# (worst adjacent pair ΔE 24.7 light / 26.8 dark, OKLab ×100).
PALETTE = {
    "light": {
        "planned": "#2a78d6",
        "measured": "#eb6834",
        "negative": "#2a78d6",
        "muted": "#898781",
        "good": "#0ca30c",
        "critical": "#d03b3b",
    },
    "dark": {
        "planned": "#3987e5",
        "measured": "#d95926",
        "negative": "#3987e5",
        "muted": "#898781",
        "good": "#0ca30c",
        "critical": "#d03b3b",
    },
}

# ── i18n ──────────────────────────────────────────────────────────────────────
I18N = {
    "en": {
        "page_title": "EV Smart Charging",
        "title": "EV Smart Charging — Operations",
        "settings": "Settings",
        "language": "Language",
        "station": "Station ID",
        "evse": "EVSE ID",
        "current_soc": "Current SoC",
        "target_soc": "Target SoC",
        "departure": "Departure (UTC)",
        "history": "History (hours)",
        "auto_refresh": "Auto-refresh every 60 s",
        "refresh_now": "Refresh now",
        "k_soc": "Current SoC",
        "k_power": "Max power",
        "k_price": "Price now",
        "k_projected": "SoC at departure",
        "k_planned_cost": "Planned cost",
        "k_measured_cost": "Measured cost",
        "d_target": "target {v}",
        "d_battery": "battery {v} kWh",
        "d_avg": "planned avg {v}",
        "d_met": "target met",
        "d_missed": "target missed",
        "d_energy": "{v} kWh",
        "d_vs_baseline": "{v} vs always-on",
        "d_unpriced": "{v} kWh unpriced",
        "d_departs": "departs in {v} h",
        "chart_price": "Spot price (€/kWh)",
        "chart_power": "Charging power (kW)",
        "chart_soc": "State of charge",
        "chart_weather": "Weather forecast (model inputs)",
        "soc_caption": "Forecast only — OCPP does not report vehicle SoC.",
        "sessions": "Recent sessions",
        "hourly": "Hourly data",
        "no_sessions": "No sessions recorded for this station.",
        "legend": "Series",
        "s_measured": "Measured (past)",
        "s_planned": "Planned (SAC)",
        "s_forced": "Forced fill (SoC guarantee)",
        "negative_hours": "Negative price hours",
        "target_line": "target {v}",
        "now_line": "now",
        "departure_line": "departure",
        "err_db": "Database unreachable — measured history and sessions are unavailable. The forecast is unaffected.",
        "err_weather": "Weather API unavailable — the model is running on fallback values.",
        "err_fetch": "Could not reach the inference server at {url}: {err}",
        "err_api": "The inference server returned {code}: {detail}",
        "sources": "Prices: **{p}** · Weather: **{w}** · Database: **{d}** · Model: `{m}` · Generated {t} UTC",
        "src_live": "live",
        "src_csv": "CSV fallback",
        "src_stale": "stale cache",
        "src_ok": "ok",
        "src_na": "unavailable",
        "src_unknown": "unknown",
        "c_hour": "Hour (UTC)",
        "c_price": "Price €/kWh",
        "c_power": "Power kW",
        "c_soc": "SoC %",
        "c_cost": "Cost €",
        "c_forced": "Forced",
        "c_phase": "Phase",
        "c_start": "Start",
        "c_end": "End",
        "c_kwh": "kWh",
        "c_state": "State",
        "c_temp": "Temperature °C",
        "c_rad": "Irradiance W/m²",
        "auth_required": "Sign in with your AI-Charge account to view the dashboard.",
        "auth_sign_in": "Sign in with Keycloak",
        "auth_sign_out": "Sign out",
        "auth_signed_in": "Signed in as {v}",
        "auth_unconfigured": "Authentication is not configured, so the dashboard will not start. Provide the Keycloak settings in .streamlit/secrets.toml (scripts/dashboard.sh writes them from Infisical).",
        "auth_anonymous": "Running without authentication (DASHBOARD_ALLOW_ANONYMOUS=1). Never use this outside local development.",
        "auth_sso_note": "Single sign-on via Keycloak",
    },
    "ja": {
        "page_title": "EV スマート充電",
        "title": "EV スマート充電 — 運用ダッシュボード",
        "settings": "設定",
        "language": "言語",
        "station": "ステーション ID",
        "evse": "EVSE ID",
        "current_soc": "現在 SoC",
        "target_soc": "目標 SoC",
        "departure": "出発時刻 (UTC)",
        "history": "実績の期間（時間）",
        "auto_refresh": "60 秒ごとに自動更新",
        "refresh_now": "今すぐ更新",
        "k_soc": "現在 SoC",
        "k_power": "最大出力",
        "k_price": "現在の価格",
        "k_projected": "出発時 SoC",
        "k_planned_cost": "計画コスト",
        "k_measured_cost": "実測コスト",
        "d_target": "目標 {v}",
        "d_battery": "バッテリ {v} kWh",
        "d_avg": "計画平均 {v}",
        "d_met": "目標達成",
        "d_missed": "目標未達",
        "d_energy": "{v} kWh",
        "d_vs_baseline": "常時充電比 {v}",
        "d_unpriced": "うち {v} kWh は価格不明",
        "d_departs": "あと {v} 時間で出発",
        "chart_price": "スポット価格 (€/kWh)",
        "chart_power": "充電電力 (kW)",
        "chart_soc": "充電率 (SoC)",
        "chart_weather": "気象予報（モデル入力）",
        "soc_caption": "予測のみ — OCPP は車両の SoC を報告しません。",
        "sessions": "直近のセッション",
        "hourly": "時間別データ",
        "no_sessions": "このステーションのセッション記録はありません。",
        "legend": "系列",
        "s_measured": "実測（過去）",
        "s_planned": "計画（SAC）",
        "s_forced": "強制充電（SoC 保証）",
        "negative_hours": "マイナス価格の時間",
        "target_line": "目標 {v}",
        "now_line": "現在",
        "departure_line": "出発",
        "err_db": "データベースに接続できません。実績とセッションは表示できません（予測は影響を受けません）。",
        "err_weather": "気象 API に接続できないため、モデルはフォールバック値で動作しています。",
        "err_fetch": "推論サーバ {url} に接続できません: {err}",
        "err_api": "推論サーバがエラーを返しました ({code}): {detail}",
        "sources": "価格: **{p}** ・ 気象: **{w}** ・ DB: **{d}** ・ モデル: `{m}` ・ 生成 {t} UTC",
        "src_live": "ライブ",
        "src_csv": "CSV フォールバック",
        "src_stale": "古いキャッシュ",
        "src_ok": "正常",
        "src_na": "利用不可",
        "src_unknown": "不明",
        "c_hour": "時刻 (UTC)",
        "c_price": "価格 €/kWh",
        "c_power": "電力 kW",
        "c_soc": "SoC %",
        "c_cost": "コスト €",
        "c_forced": "強制",
        "c_phase": "区分",
        "c_start": "開始",
        "c_end": "終了",
        "c_kwh": "kWh",
        "c_state": "状態",
        "c_temp": "気温 °C",
        "c_rad": "日射量 W/m²",
        "auth_required": "ダッシュボードを表示するには AI-Charge アカウントでサインインしてください。",
        "auth_sign_in": "Keycloak でサインイン",
        "auth_sign_out": "サインアウト",
        "auth_signed_in": "{v} としてサインイン中",
        "auth_unconfigured": "認証が未設定のためダッシュボードを起動できません。Keycloak の設定を .streamlit/secrets.toml に用意してください（scripts/dashboard.sh が Infisical から書き出します）。",
        "auth_anonymous": "認証なしで動作しています（DASHBOARD_ALLOW_ANONYMOUS=1）。ローカル開発以外では使用しないでください。",
        "auth_sso_note": "Keycloak によるシングルサインオン",
    },
    "de": {
        "page_title": "EV-Smart-Charging",
        "title": "EV-Smart-Charging — Betrieb",
        "settings": "Einstellungen",
        "language": "Sprache",
        "station": "Ladepunkt-ID",
        "evse": "EVSE-ID",
        "current_soc": "Aktueller SoC",
        "target_soc": "Ziel-SoC",
        "departure": "Abfahrt (UTC)",
        "history": "Verlauf (Stunden)",
        "auto_refresh": "Alle 60 s aktualisieren",
        "refresh_now": "Jetzt aktualisieren",
        "k_soc": "Aktueller SoC",
        "k_power": "Maximale Leistung",
        "k_price": "Preis jetzt",
        "k_projected": "SoC bei Abfahrt",
        "k_planned_cost": "Geplante Kosten",
        "k_measured_cost": "Gemessene Kosten",
        "d_target": "Ziel {v}",
        "d_battery": "Batterie {v} kWh",
        "d_avg": "geplanter Schnitt {v}",
        "d_met": "Ziel erreicht",
        "d_missed": "Ziel verfehlt",
        "d_energy": "{v} kWh",
        "d_vs_baseline": "{v} ggü. Dauerladen",
        "d_unpriced": "{v} kWh ohne Preis",
        "d_departs": "Abfahrt in {v} h",
        "chart_price": "Börsenstrompreis (€/kWh)",
        "chart_power": "Ladeleistung (kW)",
        "chart_soc": "Ladezustand (SoC)",
        "chart_weather": "Wettervorhersage (Modelleingaben)",
        "soc_caption": "Nur Prognose — OCPP meldet den Fahrzeug-SoC nicht.",
        "sessions": "Letzte Ladevorgänge",
        "hourly": "Stundenwerte",
        "no_sessions": "Für diesen Ladepunkt sind keine Ladevorgänge erfasst.",
        "legend": "Reihen",
        "s_measured": "Gemessen (Vergangenheit)",
        "s_planned": "Geplant (SAC)",
        "s_forced": "Zwangsladung (SoC-Garantie)",
        "negative_hours": "Stunden mit negativem Preis",
        "target_line": "Ziel {v}",
        "now_line": "jetzt",
        "departure_line": "Abfahrt",
        "err_db": "Datenbank nicht erreichbar — gemessener Verlauf und Ladevorgänge sind nicht verfügbar. Die Prognose ist davon nicht betroffen.",
        "err_weather": "Wetter-API nicht erreichbar — das Modell arbeitet mit Ersatzwerten.",
        "err_fetch": "Inferenzserver unter {url} nicht erreichbar: {err}",
        "err_api": "Der Inferenzserver antwortete mit {code}: {detail}",
        "sources": "Preise: **{p}** · Wetter: **{w}** · Datenbank: **{d}** · Modell: `{m}` · Erstellt {t} UTC",
        "src_live": "live",
        "src_csv": "CSV-Ersatz",
        "src_stale": "veralteter Cache",
        "src_ok": "ok",
        "src_na": "nicht verfügbar",
        "src_unknown": "unbekannt",
        "c_hour": "Stunde (UTC)",
        "c_price": "Preis €/kWh",
        "c_power": "Leistung kW",
        "c_soc": "SoC %",
        "c_cost": "Kosten €",
        "c_forced": "Zwang",
        "c_phase": "Phase",
        "c_start": "Beginn",
        "c_end": "Ende",
        "c_kwh": "kWh",
        "c_state": "Status",
        "c_temp": "Temperatur °C",
        "c_rad": "Einstrahlung W/m²",
        "auth_required": "Melden Sie sich mit Ihrem AI-Charge-Konto an, um das Dashboard zu sehen.",
        "auth_sign_in": "Mit Keycloak anmelden",
        "auth_sign_out": "Abmelden",
        "auth_signed_in": "Angemeldet als {v}",
        "auth_unconfigured": "Die Authentifizierung ist nicht konfiguriert, daher startet das Dashboard nicht. Hinterlegen Sie die Keycloak-Einstellungen in .streamlit/secrets.toml (scripts/dashboard.sh schreibt sie aus Infisical).",
        "auth_anonymous": "Läuft ohne Authentifizierung (DASHBOARD_ALLOW_ANONYMOUS=1). Außerhalb der lokalen Entwicklung niemals verwenden.",
        "auth_sso_note": "Single Sign-on über Keycloak",
    },
}

LANGUAGES = {"English": "en", "日本語": "ja", "Deutsch": "de"}


def tr(lang: str, key: str, **kwargs) -> str:
    """Looks up a translated string, falling back to English.

    Args:
        lang: Language code, ``"en"`` or ``"ja"``.
        key: Translation key.
        **kwargs: Values substituted into ``{placeholders}``.

    Returns:
        str: The translated string.
    """
    text = I18N.get(lang, I18N["en"]).get(key) or I18N["en"][key]
    return text.format(**kwargs) if kwargs else text


# ── Authentication ────────────────────────────────────────────────────────────


def auth_configured() -> bool:
    """Reports whether an OIDC provider is configured for this app.

    Streamlit reads the provider from the ``[auth]`` section of
    ``.streamlit/secrets.toml``, which ``scripts/dashboard.sh`` writes
    from Infisical at startup.

    Returns:
        bool: ``True`` if ``st.login`` can be called.
    """
    try:
        return "auth" in st.secrets and AUTH_PROVIDER in st.secrets["auth"]
    except Exception:
        return False


def require_login(lang: str) -> bool:
    """Gates the page on a Keycloak session.

    Fails closed: with no provider configured the dashboard refuses to
    render rather than serving operational data anonymously. Set
    ``DASHBOARD_ALLOW_ANONYMOUS=1`` to bypass this for local development
    only.

    Args:
        lang: Active language code, for the sign-in copy.

    Returns:
        bool: ``True`` when the caller may see the dashboard.
    """
    if ALLOW_ANONYMOUS:
        st.warning(tr(lang, "auth_anonymous"), icon="⚠️")
        return True

    if not auth_configured():
        st.error(tr(lang, "auth_unconfigured"), icon="🔒")
        return False

    if not st.user.is_logged_in:
        sign_in_page(lang)
        return False

    return True


# Signed out there is nothing to filter, so the sidebar is hidden and the page
# collapses to a single centred card — a sign-in screen rather than an empty
# dashboard carrying an alert box.
_SIGN_IN_CSS = """
<style>
  [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] { display: none; }
  [data-testid="stMainBlockContainer"] { padding-top: 9vh; }
  /* Streamlit's own bordered container is the card; styling it here keeps the
     widgets inside it, which a hand-written <div> cannot do — each st.markdown
     call is closed off in its own block. */
  [data-testid="stVerticalBlockBorderWrapper"] { border-radius: 14px; }
  [data-testid="stVerticalBlockBorderWrapper"] > div > div { padding: 6px 4px; }
  .signin-head { text-align: center; }
  .signin-head img { width: 56px; height: 56px; border-radius: 12px; }
  .signin-head .t { font-size: 1.3rem; font-weight: 620; margin: 16px 0 6px; }
  .signin-head .s { font-size: .92rem; opacity: .72; line-height: 1.5; margin: 0 0 18px; }
  .signin-foot { font-size: .78rem; opacity: .55; text-align: center; margin-top: 12px; }
</style>
"""


@st.cache_data(show_spinner=False)
def _mark_data_uri() -> str:
    """Returns the brand mark as a data URI, or an empty string if absent.

    Inlining it keeps the whole card in one markdown block, which is what
    lets the mark, heading and copy share a single centred layout.

    Returns:
        str: An ``<img>`-ready data URI, or ``""``.
    """
    if not SIGN_IN_MARK.exists():
        return ""
    return (
        "data:image/png;base64," + base64.b64encode(SIGN_IN_MARK.read_bytes()).decode()
    )


def sign_in_page(lang: str) -> None:
    """Renders the signed-out screen: a centred card, and nothing else.

    Args:
        lang: Active language code.
    """
    st.markdown(_SIGN_IN_CSS, unsafe_allow_html=True)
    _, middle, _ = st.columns([1, 1.15, 1])

    with middle:
        with st.container(border=True):
            mark = _mark_data_uri()
            st.markdown(
                '<div class="signin-head">'
                + (f'<img src="{mark}" alt="">' if mark else "")
                + f'<div class="t">{tr(lang, "page_title")}</div>'
                + f'<div class="s">{tr(lang, "auth_required")}</div>'
                + "</div>",
                unsafe_allow_html=True,
            )
            st.button(
                tr(lang, "auth_sign_in"),
                type="primary",
                width="stretch",
                on_click=st.login,
                args=(AUTH_PROVIDER,),
            )

        # The sidebar is hidden here, so the language control has to live on the
        # sign-in screen or the other two languages are unreachable signed out.
        inner = st.columns([1, 1.4, 1])[1]
        with inner:
            picked = st.selectbox(
                I18N[lang]["language"],
                list(LANGUAGES),
                index=list(LANGUAGES.values()).index(lang),
                key="lang_choice_signin",
                label_visibility="collapsed",
            )
        if LANGUAGES[picked] != lang:
            st.session_state["lang"] = LANGUAGES[picked]
            st.rerun()

        st.markdown(
            f'<div class="signin-foot">{tr(lang, "auth_sso_note")}</div>',
            unsafe_allow_html=True,
        )


def account_controls(lang: str) -> None:
    """Shows who is signed in, and a sign-out button, in the sidebar.

    Args:
        lang: Active language code.
    """
    if ALLOW_ANONYMOUS or not auth_configured():
        return
    with st.sidebar:
        who = getattr(st.user, "name", None) or getattr(st.user, "email", None) or "?"
        st.caption(tr(lang, "auth_signed_in", v=who))
        st.button(tr(lang, "auth_sign_out"), on_click=st.logout)


# ── Data ──────────────────────────────────────────────────────────────────────


@st.cache_data(ttl=30, show_spinner=False)
def fetch_payload(params: tuple) -> dict:
    """Calls ``GET /api/dashboard`` on the inference server.

    Args:
        params: Query parameters as a tuple of ``(key, value)`` pairs,
            so the result is cacheable.

    Returns:
        dict: The decoded dashboard payload.

    Raises:
        requests.HTTPError: If the server answers with an error status.
        requests.RequestException: If the server cannot be reached.
    """
    headers = {"Authorization": f"Bearer {API_TOKEN}"} if API_TOKEN else {}
    response = requests.get(
        f"{SERVE_URL}/api/dashboard",
        params=dict(params),
        headers=headers,
        timeout=REQUEST_TIMEOUT_S,
    )
    response.raise_for_status()
    return response.json()


def theme_colors() -> dict:
    """Returns the palette matching the viewer's active Streamlit theme.

    Returns:
        dict: The ``light`` or ``dark`` entry of :data:`PALETTE`.
    """
    try:
        mode = st.context.theme.type
    except Exception:
        mode = None
    return PALETTE["dark" if mode == "dark" else "light"]


# ── Charts ────────────────────────────────────────────────────────────────────


def _time_axis(df: pd.DataFrame, lang: str) -> alt.X:
    """Builds the x encoding every timeline shares.

    Pinning the domain to the full frame keeps the price, power and SoC
    charts on one axis, so an hour sits at the same place in all three
    even when a chart has no data for it.

    Args:
        df: Frame from :func:`dashboard.merge_timeline`.
        lang: Active language code.

    Returns:
        altair.X: The shared x encoding.
    """
    return alt.X(
        "t:T",
        title=None,
        axis=alt.Axis(format="%H:%M", grid=False),
        scale=alt.Scale(domain=[df["t"].min(), df["t_end"].max()]),
    )


def _negative_price_layer(df: pd.DataFrame, colors: dict):
    """Tints the hours whose spot price is negative.

    Drawn behind every timeline so the same hours line up across charts.
    Negative prices are the behaviour the agent is trained to exploit,
    so they are called out rather than left for the reader to spot.

    Args:
        df: Frame from :func:`dashboard.merge_timeline`.
        colors: Active palette.

    Returns:
        altair.Chart: The background layer (empty if no negative hours).
    """
    neg = df[df["price_eur_kwh"] < 0]
    return (
        alt.Chart(neg)
        .mark_rect(opacity=0.12, color=colors["negative"])
        .encode(x="t:T", x2="t_end:T")
    )


def _now_rule(now: datetime, label: str, colors: dict):
    """Draws the vertical "now" marker shared by every timeline.

    Args:
        now: The boundary between measured and forecast hours.
        label: Text to place beside the rule.
        colors: Active palette.

    Returns:
        altair.LayerChart: Rule plus its label.
    """
    frame = pd.DataFrame({"t": [now], "label": [label]})
    rule = alt.Chart(frame).mark_rule(color=colors["muted"], size=1).encode(x="t:T")
    text = (
        alt.Chart(frame)
        .mark_text(align="left", dx=4, baseline="top", color=colors["muted"])
        .encode(x="t:T", y=alt.value(6), text="label:N")
    )
    return rule + text


def price_chart(df: pd.DataFrame, now: datetime, lang: str, colors: dict):
    """Builds the spot-price timeline.

    Args:
        df: Frame from :func:`dashboard.merge_timeline`.
        now: Boundary between measured and forecast hours.
        lang: Active language code.
        colors: Active palette.

    Returns:
        altair.LayerChart: The price chart.
    """
    priced = df.dropna(subset=["price_eur_kwh"])
    line = (
        alt.Chart(priced)
        .mark_line(color=colors["planned"], strokeWidth=2, interpolate="step-after")
        .encode(
            x=_time_axis(df, lang),
            y=alt.Y("price_eur_kwh:Q", title=tr(lang, "c_price")),
            tooltip=[
                alt.Tooltip("t:T", title=tr(lang, "c_hour"), format="%b %d %H:%M"),
                alt.Tooltip("price_eur_kwh:Q", title=tr(lang, "c_price"), format=".4f"),
            ],
        )
    )
    zero = (
        alt.Chart(pd.DataFrame({"y": [0]}))
        .mark_rule(color=colors["muted"], size=1)
        .encode(y="y:Q")
    )
    layers = [
        _negative_price_layer(df, colors),
        zero,
        line,
        _now_rule(now, tr(lang, "now_line"), colors),
    ]
    return alt.layer(*layers).properties(height=190).interactive(bind_y=False)


def power_chart(df: pd.DataFrame, now: datetime, lang: str, colors: dict):
    """Builds the measured-vs-planned charging power timeline.

    Args:
        df: Frame from :func:`dashboard.merge_timeline`.
        now: Boundary between measured and forecast hours.
        lang: Active language code.
        colors: Active palette.

    Returns:
        altair.LayerChart: The power chart.
    """
    labels = {
        "measured": tr(lang, "s_measured"),
        "planned": tr(lang, "s_planned"),
        "forced": tr(lang, "s_forced"),
    }
    data = df[df["power_kw"] > 0].assign(series_label=lambda d: d["series"].map(labels))
    bars = (
        alt.Chart(data)
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=_time_axis(df, lang),
            x2="t_end:T",
            y=alt.Y("power_kw:Q", title=tr(lang, "c_power")),
            # An explicit zero baseline: giving x2 turns off the implicit
            # one, which otherwise collapses every bar to a hairline.
            y2=alt.Y2(datum=0),
            color=alt.Color(
                "series_label:N",
                title=tr(lang, "legend"),
                scale=alt.Scale(
                    domain=[labels["measured"], labels["planned"], labels["forced"]],
                    range=[colors["measured"], colors["planned"], colors["planned"]],
                ),
                legend=alt.Legend(orient="top", labelLimit=320),
            ),
            # Forced-fill hours share the planned hue; opacity is the
            # second channel so they never rely on colour alone.
            opacity=alt.Opacity(
                "series:N",
                scale=alt.Scale(
                    domain=["measured", "planned", "forced"], range=[1.0, 1.0, 0.55]
                ),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("t:T", title=tr(lang, "c_hour"), format="%b %d %H:%M"),
                alt.Tooltip("series_label:N", title=tr(lang, "legend")),
                alt.Tooltip("power_kw:Q", title=tr(lang, "c_power"), format=".2f"),
                alt.Tooltip("price_eur_kwh:Q", title=tr(lang, "c_price"), format=".4f"),
                alt.Tooltip("cost_eur:Q", title=tr(lang, "c_cost"), format=".4f"),
            ],
        )
    )
    layers = [
        _negative_price_layer(df, colors),
        bars,
        _now_rule(now, tr(lang, "now_line"), colors),
    ]
    return alt.layer(*layers).properties(height=210).interactive(bind_y=False)


def soc_chart(df: pd.DataFrame, payload: dict, now: datetime, lang: str, colors: dict):
    """Builds the projected state-of-charge trajectory.

    Args:
        df: Frame from :func:`dashboard.merge_timeline`.
        payload: The dashboard payload (for the target and departure).
        now: Boundary between measured and forecast hours.
        lang: Active language code.
        colors: Active palette.

    Returns:
        altair.LayerChart: The SoC chart.
    """
    trajectory = df.dropna(subset=["soc"])
    line = (
        alt.Chart(trajectory)
        .mark_line(color=colors["planned"], strokeWidth=2, point=False)
        .encode(
            x=_time_axis(df, lang),
            y=alt.Y(
                "soc:Q",
                title=tr(lang, "c_soc"),
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format="%"),
            ),
            tooltip=[
                alt.Tooltip("t:T", title=tr(lang, "c_hour"), format="%b %d %H:%M"),
                alt.Tooltip("soc:Q", title=tr(lang, "c_soc"), format=".1%"),
            ],
        )
    )
    target = payload["target_soc"]
    target_frame = pd.DataFrame(
        {
            "y": [target],
            "t": [df["t"].min()],
            "label": [tr(lang, "target_line", v=f"{target:.0%}")],
        }
    )
    # Dashed because it is a threshold, not a gridline.
    target_rule = (
        alt.Chart(target_frame)
        .mark_rule(color=colors["muted"], strokeDash=[5, 4], size=1.5)
        .encode(y="y:Q")
    )
    target_text = (
        alt.Chart(target_frame)
        .mark_text(align="left", dx=4, dy=-6, color=colors["muted"])
        .encode(y="y:Q", x="t:T", text="label:N")
    )
    departure = pd.to_datetime(payload["departure_time"])
    dep_frame = pd.DataFrame({"t": [departure], "label": [tr(lang, "departure_line")]})
    dep_rule = (
        alt.Chart(dep_frame)
        .mark_rule(color=colors["muted"], strokeDash=[3, 3], size=1)
        .encode(x="t:T")
    )
    dep_text = (
        alt.Chart(dep_frame)
        .mark_text(align="right", dx=-4, baseline="top", color=colors["muted"])
        .encode(x="t:T", y=alt.value(6), text="label:N")
    )
    layers = [
        _negative_price_layer(df, colors),
        target_rule,
        target_text,
        dep_rule,
        dep_text,
        line,
        _now_rule(now, tr(lang, "now_line"), colors),
    ]
    return alt.layer(*layers).properties(height=190).interactive(bind_y=False)


def weather_chart(df: pd.DataFrame, column: str, title: str, lang: str, colors: dict):
    """Builds a compact single-measure weather sparkline.

    Args:
        df: Frame from :func:`dashboard.merge_timeline`.
        column: ``temp_c`` or ``radiation_wm2``.
        title: Y-axis title.
        lang: Active language code.
        colors: Active palette.

    Returns:
        altair.Chart: The sparkline.
    """
    data = df.dropna(subset=[column])
    return (
        alt.Chart(data)
        .mark_area(
            color=colors["planned"],
            opacity=0.18,
            line={"color": colors["planned"], "strokeWidth": 2},
        )
        .encode(
            x=_time_axis(df, lang),
            y=alt.Y(f"{column}:Q", title=title),
            tooltip=[
                alt.Tooltip("t:T", title=tr(lang, "c_hour"), format="%b %d %H:%M"),
                alt.Tooltip(f"{column}:Q", title=title, format=".1f"),
            ],
        )
        .properties(height=120)
    )


# ── Page ──────────────────────────────────────────────────────────────────────


def sidebar() -> tuple[str, dict, bool]:
    """Renders the control sidebar.

    Returns:
        tuple: ``(lang, query_params, auto_refresh)``.
    """
    default_lang = st.session_state.get("lang", "en")
    with st.sidebar:
        # Keying the widget means Streamlit restores its value *before* this
        # line runs, so the control's own label is already in the newly
        # chosen language rather than lagging one interaction behind.
        chosen = st.session_state.get("lang_choice")
        label_lang = LANGUAGES.get(chosen, default_lang)
        # A dropdown rather than a radio row: the list grows with each
        # language and a horizontal radio stops fitting the sidebar.
        label = st.selectbox(
            I18N[label_lang]["language"],
            list(LANGUAGES),
            index=list(LANGUAGES.values()).index(default_lang),
            key="lang_choice",
        )
        lang = LANGUAGES[label]
        st.session_state["lang"] = lang

        st.header(tr(lang, "settings"))
        station_id = st.text_input(
            tr(lang, "station"), value=os.getenv("DASHBOARD_STATION_ID", "")
        )
        evse_id = st.number_input(
            tr(lang, "evse"),
            min_value=0,
            step=1,
            value=int(os.getenv("DASHBOARD_EVSE_ID", "1")),
        )
        current_soc = st.slider(tr(lang, "current_soc"), 0.0, 1.0, 0.20, 0.01)
        desired_soc = st.slider(tr(lang, "target_soc"), 0.0, 1.0, 0.80, 0.01)

        default_dep = (
            datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
            + timedelta(hours=12)
        ).replace(tzinfo=None)
        dep_date = st.date_input(tr(lang, "departure"), value=default_dep.date())
        dep_time = st.time_input(
            tr(lang, "departure"),
            value=default_dep.time(),
            label_visibility="collapsed",
        )
        history_hours = st.slider(tr(lang, "history"), 6, 72, 24, 1)

        auto_refresh = st.toggle(tr(lang, "auto_refresh"), value=True)
        if st.button(tr(lang, "refresh_now"), width="stretch"):
            st.cache_data.clear()

    params = {
        "station_id": station_id,
        "evse_id": int(evse_id),
        "current_soc": current_soc,
        "desired_soc": desired_soc,
        "departure_time": datetime.combine(dep_date, dep_time).isoformat(),
        "history_hours": history_hours,
    }
    return lang, params, auto_refresh


def render(payload: dict, lang: str) -> None:
    """Renders the whole dashboard body from one payload.

    Args:
        payload: A response from ``GET /api/dashboard``.
        lang: Active language code.
    """
    colors = theme_colors()
    df = dashboard.merge_timeline(payload)
    now = pd.to_datetime(payload["generated_at"])
    kpi = payload["kpi"]

    for warning in payload["warnings"]:
        if warning.startswith("database_unreachable"):
            st.warning(tr(lang, "err_db"), icon="⚠️")
        elif warning == "weather_unavailable":
            st.warning(tr(lang, "err_weather"), icon="⚠️")

    def kpi_tile(column, label, value, note, tone=None):
        """One stat tile: the number on the metric, its note underneath.

        ``st.metric``'s delta slot always draws a trend arrow, which would
        read as a change these notes do not describe — so the note goes in
        a caption instead.
        """
        with column:
            st.metric(label, value)
            if tone:
                st.markdown(
                    f"<span style='color:{tone};font-size:.82rem'>{note}</span>",
                    unsafe_allow_html=True,
                )
            else:
                st.caption(note)

    cols = st.columns(6)
    kpi_tile(
        cols[0],
        tr(lang, "k_soc"),
        f"{payload['current_soc']:.0%}",
        tr(lang, "d_target", v=f"{payload['target_soc']:.0%}"),
    )
    kpi_tile(
        cols[1],
        tr(lang, "k_power"),
        f"{payload['max_power_kw']:.1f} kW",
        tr(lang, "d_battery", v=f"{payload['battery_capacity_kwh']:.0f}"),
    )
    kpi_tile(
        cols[2],
        tr(lang, "k_price"),
        f"€{payload['forecast'][0]['price_eur_kwh']:.4f}",
        tr(lang, "d_avg", v=f"€{kpi['avg_price_eur_kwh']:.4f}"),
    )
    met = kpi["soc_target_met"]
    kpi_tile(
        cols[3],
        tr(lang, "k_projected"),
        f"{kpi['projected_soc_at_departure']:.0%}",
        ("✓ " if met else "✕ ") + tr(lang, "d_met" if met else "d_missed"),
        tone=colors["good"] if met else colors["critical"],
    )
    saving = kpi["savings_eur"]
    kpi_tile(
        cols[4],
        tr(lang, "k_planned_cost"),
        f"€{kpi['planned_cost_eur']:.2f}",
        tr(
            lang,
            "d_vs_baseline",
            v=("−" if saving >= 0 else "+") + f"€{abs(saving):.2f}",
        ),
    )
    measured_note = tr(lang, "d_energy", v=f"{kpi['actual_energy_kwh']:.1f}")
    if kpi["actual_unpriced_kwh"] > 0:
        measured_note += " · " + tr(
            lang, "d_unpriced", v=f"{kpi['actual_unpriced_kwh']:.1f}"
        )
    kpi_tile(
        cols[5],
        tr(lang, "k_measured_cost"),
        f"€{kpi['actual_cost_eur']:.2f}",
        measured_note,
    )

    st.subheader(tr(lang, "chart_price"))
    st.markdown(
        "<span style='font-size:.82rem;color:#898781'>"
        f"<span style='display:inline-block;width:10px;height:10px;border-radius:2px;"
        f"background:{colors['negative']};opacity:.35;margin-right:6px'></span>"
        f"{tr(lang, 'negative_hours')}</span>",
        unsafe_allow_html=True,
    )
    st.altair_chart(price_chart(df, now, lang, colors), width="stretch")

    st.subheader(tr(lang, "chart_power"))
    st.altair_chart(power_chart(df, now, lang, colors), width="stretch")

    st.subheader(tr(lang, "chart_soc"))
    st.caption(tr(lang, "soc_caption"))
    st.altair_chart(soc_chart(df, payload, now, lang, colors), width="stretch")

    st.subheader(tr(lang, "chart_weather"))
    left, right = st.columns(2)
    with left:
        st.altair_chart(
            weather_chart(df, "temp_c", tr(lang, "c_temp"), lang, colors),
            width="stretch",
        )
    with right:
        st.altair_chart(
            weather_chart(df, "radiation_wm2", tr(lang, "c_rad"), lang, colors),
            width="stretch",
        )

    st.subheader(tr(lang, "sessions"))
    if payload["sessions"]:
        sessions = pd.DataFrame(payload["sessions"])[
            ["start", "end", "total_kwh", "charging_state"]
        ]
        for column in ("start", "end"):
            sessions[column] = pd.to_datetime(sessions[column]).dt.strftime(
                "%Y-%m-%d %H:%M"
            )
        sessions.columns = [
            tr(lang, "c_start"),
            tr(lang, "c_end"),
            tr(lang, "c_kwh"),
            tr(lang, "c_state"),
        ]
        st.dataframe(sessions, hide_index=True, width="stretch")
    else:
        st.info(tr(lang, "no_sessions"))

    # The table view is the WCAG-clean twin of the charts above.
    with st.expander(tr(lang, "hourly")):
        table = df[
            ["t", "phase", "price_eur_kwh", "power_kw", "soc", "cost_eur", "forced"]
        ].copy()
        table["t"] = table["t"].dt.strftime("%Y-%m-%d %H:%M")
        table["soc"] = table["soc"] * 100.0
        table.columns = [
            tr(lang, "c_hour"),
            tr(lang, "c_phase"),
            tr(lang, "c_price"),
            tr(lang, "c_power"),
            tr(lang, "c_soc"),
            tr(lang, "c_cost"),
            tr(lang, "c_forced"),
        ]
        st.dataframe(table, hide_index=True, width="stretch")

    src = payload["data_sources"]
    names = {
        "live": "src_live",
        "csv": "src_csv",
        "stale-cache": "src_stale",
        "ok": "src_ok",
        "unavailable": "src_na",
    }
    st.caption(
        tr(
            lang,
            "sources",
            p=tr(lang, names.get(src["prices"], "src_unknown")),
            w=tr(lang, names.get(src["weather"], "src_unknown")),
            d=tr(lang, names.get(src["db"], "src_unknown")),
            m=payload["model"],
            t=now.strftime("%Y-%m-%d %H:%M"),
        )
    )


def main() -> None:
    """Entry point: authenticates, then renders the page and auto-refresh."""
    lang = st.session_state.get("lang", "en")
    if not require_login(lang):
        return

    lang, params, auto_refresh = sidebar()
    account_controls(lang)
    st.title(tr(lang, "title"))

    @st.fragment(run_every=60 if auto_refresh else None)
    def body():
        try:
            payload = fetch_payload(tuple(sorted(params.items())))
        except requests.HTTPError as exc:
            detail = ""
            try:
                detail = exc.response.json().get("detail", "")
            except Exception:
                detail = exc.response.text[:200]
            st.error(
                tr(lang, "err_api", code=exc.response.status_code, detail=detail),
                icon="🚫",
            )
            return
        except requests.RequestException as exc:
            st.error(tr(lang, "err_fetch", url=SERVE_URL, err=exc), icon="🚫")
            return
        render(payload, lang)

    body()


st.set_page_config(
    page_title=I18N[st.session_state.get("lang", "en")]["page_title"],
    # The brand mark, committed under assets/ so nothing is fetched at runtime.
    page_icon=str(PAGE_ICON) if PAGE_ICON.exists() else "⚡",
    layout="wide",
)

if __name__ == "__main__":
    main()
