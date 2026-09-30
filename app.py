import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.graph_objects as go
from pathlib import Path

APP_DIR = Path(__file__).parent

st.set_page_config(page_title="Football IQ", page_icon="⚽", layout="wide")

# ---------- load artifacts ----------
@st.cache_resource
def load_match_artifacts():
    result_model = joblib.load(APP_DIR / "result_model.joblib")
    corner_model_home = joblib.load(APP_DIR / "corner_model_home.joblib")
    corner_model_away = joblib.load(APP_DIR / "corner_model_away.joblib")
    foul_model_home = joblib.load(APP_DIR / "foul_model_home.joblib")
    foul_model_away = joblib.load(APP_DIR / "foul_model_away.joblib")
    le = joblib.load(APP_DIR / "label_encoder.joblib")
    matches = pd.read_csv(APP_DIR / "matches_full.csv", parse_dates=["Date"])
    with open(APP_DIR / "final_elo.json") as f:
        final_elo = json.load(f)
    with open(APP_DIR / "feature_cols.json") as f:
        feature_cols = json.load(f)
    with open(APP_DIR / "teams.json") as f:
        teams = json.load(f)
    return (result_model, corner_model_home, corner_model_away, foul_model_home,
            foul_model_away, le, matches, final_elo, feature_cols, teams)


@st.cache_resource
def load_player_artifacts():
    """player_profiles.csv has one row per Player+season+POSITION LABEL (e.g. RW / FW / RW,FW),
    which splits one season into several partial rows. Recombine to one row per player-season."""
    P90 = ['Gls_p90','Ast_p90','Sh_p90','SoT_p90','Tkl_p90','Int_p90','Touches_p90','PrgP_p90','PrgC_p90']
    raw = pd.read_csv(APP_DIR / "player_profiles.csv")
    raw["total_minutes"] = raw["total_minutes"].fillna(0)

    def combine(keys):
        w = raw["total_minutes"].clip(lower=0)
        tmp = raw.copy()
        for c in P90:                                   # minutes-weighted mean == exact per-90 from totals
            tmp[c] = tmp[c].fillna(0) * w
        g = tmp.groupby(keys)
        out = g[["matches_played","total_minutes","total_goals","total_assists"] + P90].sum()
        for c in P90:
            out[c] = out[c] / out["total_minutes"].clip(lower=1)
        out["avg_market_value"] = g["avg_market_value"].mean()
        top = (raw.sort_values("total_minutes").drop_duplicates(keys, keep="last")
                  .set_index(keys)[[c for c in ["Pos","team"] if c not in keys]])
        top["Pos"] = top["Pos"].astype(str).str.split(",").str[0]
        return out.join(top).reset_index()

    by_season = combine(["Player","season"])
    by_team = combine(["Player","season","team"])
    style = (raw.dropna(subset=["cluster_name"]).sort_values("season")
                .drop_duplicates("Player", keep="last").set_index("Player")["cluster_name"])
    by_season["cluster_name"] = by_season["Player"].map(style)      # latest known playing style
    ref = by_season[(by_season["total_minutes"] >= 450) & (by_season["Pos"] != "GK")]
    return by_season, by_team, ref[P90].mean()


@st.cache_resource
def load_sentiment_artifacts():
    tfidf = joblib.load(APP_DIR / "tfidf.joblib")
    clf = joblib.load(APP_DIR / "sentiment_clf.joblib")
    timeline = pd.read_csv(APP_DIR / "sentiment_timeline.csv")
    return tfidf, clf, timeline


(result_model, corner_model_home, corner_model_away, foul_model_home, foul_model_away,
 le, matches, final_elo, feature_cols, teams) = load_match_artifacts()
player_profile, player_by_team, league_avg = load_player_artifacts()
tfidf, sentiment_clf, sentiment_timeline = load_sentiment_artifacts()

p90_cols = ['Gls_p90','Ast_p90','Sh_p90','SoT_p90','Tkl_p90','Int_p90','Touches_p90','PrgP_p90','PrgC_p90']

# ---------- shared helper functions ----------
def get_recent_form(team, matches_df, n=5):
    home = matches_df[matches_df["home_team"] == team][
        ["Date","home_goals","away_goals","home_Corners","away_Corners","home_Fouls"]
    ].rename(columns={"home_goals":"goals_for","away_goals":"goals_against",
                       "home_Corners":"corners_for","away_Corners":"corners_against",
                       "home_Fouls":"fouls_committed"})
    away = matches_df[matches_df["away_team"] == team][
        ["Date","away_goals","home_goals","away_Corners","home_Corners","away_Fouls"]
    ].rename(columns={"away_goals":"goals_for","home_goals":"goals_against",
                       "away_Corners":"corners_for","home_Corners":"corners_against",
                       "away_Fouls":"fouls_committed"})
    games = pd.concat([home, away], ignore_index=True).sort_values("Date", ascending=False).head(n)
    if len(games) == 0:
        return dict(points=1.0, goals_for=1.0, goals_against=1.0, corners_for=5.0, corners_against=5.0, fouls_committed=10.0)
    pts = np.select([games["goals_for"]>games["goals_against"], games["goals_for"]==games["goals_against"]], [3,1], default=0)
    return dict(points=pts.mean(), goals_for=games["goals_for"].mean(), goals_against=games["goals_against"].mean(),
                corners_for=games["corners_for"].mean(), corners_against=games["corners_against"].mean(),
                fouls_committed=games["fouls_committed"].mean())


def get_h2h(home_team, away_team, matches_df, n=3):
    prior = matches_df[((matches_df["home_team"]==home_team)&(matches_df["away_team"]==away_team)) |
                        ((matches_df["home_team"]==away_team)&(matches_df["away_team"]==home_team))
                       ].sort_values("Date", ascending=False).head(n)
    if len(prior)==0: return 1.0
    pts=[]
    for _, p in prior.iterrows():
        if p["home_team"]==home_team: pts.append(3 if p["result"]=="H" else (1 if p["result"]=="D" else 0))
        else: pts.append(3 if p["result"]=="A" else (1 if p["result"]=="D" else 0))
    return float(np.mean(pts))


def build_feature_row(home_team, away_team):
    hf, af = get_recent_form(home_team, matches), get_recent_form(away_team, matches)
    h_elo, a_elo = final_elo.get(home_team, 1500), final_elo.get(away_team, 1500)
    h2h = get_h2h(home_team, away_team, matches)
    row = pd.DataFrame([{
        "home_roll5_points": hf["points"], "home_roll5_goals_for": hf["goals_for"], "home_roll5_goals_against": hf["goals_against"],
        "away_roll5_points": af["points"], "away_roll5_goals_for": af["goals_for"], "away_roll5_goals_against": af["goals_against"],
        "home_roll5_corners_for": hf["corners_for"], "home_roll5_corners_against": hf["corners_against"],
        "away_roll5_corners_for": af["corners_for"], "away_roll5_corners_against": af["corners_against"],
        "home_roll5_fouls_committed": hf["fouls_committed"], "away_roll5_fouls_committed": af["fouls_committed"],
        "elo_diff": h_elo - a_elo, "h2h_home_points": h2h,
    }])[feature_cols]
    return row, h_elo, a_elo


import re, difflib, unicodedata

def _norm(x):
    x = unicodedata.normalize("NFKD", str(x)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z ]", "", x.lower()).strip()

def find_players(query, names, n=8):
    """exact -> substring -> all-words -> fuzzy, so 'Son Hung man' still finds Son Heung-min"""
    m = {_norm(p): p for p in names}; q = _norm(query)
    if not q: return []
    if q in m: return [m[q]]
    hit = [o for k, o in m.items() if q in k] or [o for k, o in m.items() if all(t in k for t in q.split())]
    if hit: return hit[:n]
    close = difflib.get_close_matches(q, list(m), n=n, cutoff=0.5)
    if not close:
        surn = {k.split()[-1]: k for k in m}
        for t in q.split(): close += [surn[x] for x in difflib.get_close_matches(t, list(surn), n=n, cutoff=0.7)]
    return [m[c] for c in close][:n]

GOALS = [{"minute": 16, "scorer": "Enner Valencia", "how": "penalty"},
         {"minute": 31, "scorer": "Enner Valencia", "how": "header"}]
MATCH = "Qatar 0-2 Ecuador (World Cup opener, 20 Nov 2022)"

def clean_tweet(t):
    t = re.sub(r"http\S+|@\w+", " ", str(t)); t = t.replace("#", "")
    return re.sub(r"\s+", " ", t).strip()

def classify(text):
    t = clean_tweet(text); vec = tfidf.transform([t])
    if not t: return None, None
    if vec.nnz == 0: return "neutral", None
    p = sentiment_clf.predict_proba(vec)[0]; i = int(np.argmax(p))
    return sentiment_clf.classes_[i], float(p[i])

def mood_word(x): return "positive" if x > .15 else "negative" if x < -.15 else "mixed / neutral"

def timeline_fig(zoom=False):
    tl = sentiment_timeline
    fig = go.Figure()
    fig.add_trace(go.Bar(x=tl['minute_bucket'], y=tl['tweet_volume'], name='Tweet volume', marker_color='#1976d2', yaxis='y1'))
    fig.add_trace(go.Scatter(x=tl['minute_bucket'], y=tl['avg_sentiment'], name='Avg sentiment', mode='lines+markers',
                             marker_color='#c62828', yaxis='y2'))
    for g in GOALS:
        fig.add_vline(x=g['minute'], line_dash="dash", line_color="#2e7d32", annotation_text=f"Goal ({g['minute']}')")
    fig.add_vline(x=0, line_color="black", opacity=.4, annotation_text="Kickoff")
    fig.update_layout(height=420, yaxis=dict(title="Tweet volume"),
                      yaxis2=dict(title="Avg sentiment", overlaying='y', side='right', range=[-1, 1]),
                      xaxis=dict(title="Minutes from kickoff", range=[-15, 50] if zoom else None),
                      legend=dict(orientation="h", y=1.1))
    return fig

def bot_reply(text, pending):
    """-> (reply, chart_key, new_pending). Never raises on odd input."""
    tl = sentiment_timeline
    q = re.sub(r"[^a-z\s]", "", str(text or "").lower()).strip()
    if not q: return "Try **goal**, **who scored** or **sentiment** (or **help**).", None, pending
    if q in {"yes","y","yeah","yep","yup","ok","okay","sure","please"}:
        if pending == "chart": return "Here is the chart zoomed on the first 50 minutes.", "zoom", None
        if pending == "goals": return bot_reply("goal", None)
        return "Yes to what? Try **goal**, **who scored** or **sentiment**.", None, None
    if q in {"no","n","nope","nah"}: return "No problem. Ask me anything else.", None, None
    if q in {"help","menu","commands"}:
        return ("Ask me: **goal**, **who scored**, **score**, **sentiment**, **busiest moment**, **accuracy**. "
                "Or type any fan tweet and I will read its mood."), None, pending
    if re.search(r"\b(who|scorer|scored|scorers)\b", q):
        m = " and ".join(f"{g['minute']}' ({g['how']})" for g in GOALS)
        return f"**Enner Valencia** scored both of Ecuador's goals: {m}.\n\nWant to see how fans reacted? (yes/no)", None, "goals"
    if re.search(r"\bgoals?\b", q):
        lines = [f"{MATCH}: **{len(GOALS)} goals**, both for Ecuador."]
        for g in GOALS:
            sub = tl[(tl.minute_bucket >= g['minute'] - 5) & (tl.minute_bucket <= g['minute'] + 5)]
            lines.append(f"- {g['minute']}' {g['scorer']} ({g['how']}): fan mood **{mood_word(sub.avg_sentiment.mean())}** "
                         f"(avg {sub.avg_sentiment.mean():+.2f}, {int(sub.tweet_volume.sum())} tweets nearby)" if len(sub)
                         else f"- {g['minute']}' {g['scorer']}: no tweets in that window")
        lines.append("\nWant the timeline chart? (yes/no)")
        return "\n".join(lines), None, "chart"
    if re.search(r"\b(score|result|final|who won|who wins)\b", q): return f"{MATCH}. Ecuador won 2-0.", None, pending
    if re.search(r"\b(sentiment|mood|feel|feeling|fans?|reaction|overall)\b", q):
        avg = np.average(tl.avg_sentiment, weights=tl.tweet_volume)
        return f"Across the match window fans were **{mood_word(avg)}** overall (avg {avg:+.2f}).\n\nWant the timeline chart? (yes/no)", None, "chart"
    if re.search(r"\b(busiest|peak|most|spike|volume)\b", q):
        r = tl.loc[tl.tweet_volume.idxmax()]
        return f"Busiest 5 minutes: {r.minute_bucket:.0f} to {r.minute_bucket+5:.0f} min from kickoff, {int(r.tweet_volume)} tweets.", None, pending
    if re.search(r"\b(accuracy|accurate|model|f1)\b", q): return "The classifier scored **71.7%** on held-out labelled tweets.", None, pending
    label, conf = classify(text)
    if label is None: return "I could not read that. Type **help**.", None, pending
    return (f"Fan mood of that message: **{label.upper()}**" + (f" ({conf:.0%} confident)" if conf else " (no words I recognise)")), None, pending

# ================= UI =================
st.title("⚽ Football IQ")
st.caption("Match prediction, player analytics, and fan sentiment — Premier League 2021-2024")

tab1, tab2, tab3 = st.tabs(["🔮 Match Predictor", "👤 Player Dashboard", "💬 Sentiment Tracker"])

# ---------------- TAB 1: MATCH PREDICTOR ----------------
with tab1:
    st.subheader("Predict a match")
    c1, c2 = st.columns(2)
    with c1:
        home_team = st.selectbox("🏠 Home team", teams, index=teams.index("Arsenal") if "Arsenal" in teams else 0, key="home_sel")
    with c2:
        away_options = [t for t in teams if t != home_team]
        away_team = st.selectbox("✈️ Away team", away_options, index=0, key="away_sel")

    if st.button("Predict Result", type="primary"):
        row, h_elo, a_elo = build_feature_row(home_team, away_team)

        proba = result_model.predict_proba(row)[0]
        proba_map = dict(zip(le.classes_, proba))
        p_home, p_draw, p_away = proba_map.get("H",0), proba_map.get("D",0), proba_map.get("A",0)

        pred_hc = corner_model_home.predict(row)[0]
        pred_ac = corner_model_away.predict(row)[0]
        pred_hf = foul_model_home.predict(row)[0]
        pred_af = foul_model_away.predict(row)[0]

        st.markdown("---")
        st.subheader(f"{home_team} vs {away_team}")

        fig = go.Figure(go.Bar(
            x=[p_home, p_draw, p_away],
            y=[f"{home_team} Win", "Draw", f"{away_team} Win"],
            orientation="h",
            marker_color=["#2e7d32", "#9e9e9e", "#c62828"],
            text=[f"{p_home:.0%}", f"{p_draw:.0%}", f"{p_away:.0%}"],
            textposition="outside",
        ))
        fig.update_layout(xaxis=dict(range=[0,1], tickformat=".0%"), height=260, margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig)

        most_likely = max(proba_map, key=proba_map.get)
        label_map = {"H": f"{home_team} Win", "D": "Draw", "A": f"{away_team} Win"}
        st.success(f"**Most likely outcome: {label_map[most_likely]}** ({proba_map[most_likely]:.0%} confidence)")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric(f"{home_team} corners", f"{pred_hc:.1f}")
        m2.metric(f"{away_team} corners", f"{pred_ac:.1f}")
        m3.metric(f"{home_team} fouls", f"{pred_hf:.1f}")
        m4.metric(f"{away_team} fouls", f"{pred_af:.1f}")

        with st.expander("See the numbers behind this prediction"):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**{home_team} (home)** — ELO {h_elo:.0f}")
            with c2:
                st.markdown(f"**{away_team} (away)** — ELO {a_elo:.0f}")

    st.caption(
        "Model: XGBoost classifier (result) + Random Forest regressors (corners, fouls), "
        "trained on 2021-24 Premier League data using rolling form, ELO rating, and "
        "head-to-head history. Held-out test accuracy ~55% (vs ~47% baseline)."
    )

# ---------------- TAB 2: PLAYER DASHBOARD ----------------
with tab2:
    st.subheader("Player profile")
    all_players = sorted(player_profile['Player'].dropna().unique())
    q = st.text_input("Type a name (typos are OK)", placeholder="e.g. Salah, Son Hung man, Alexander-Arnold", key="pq")
    hits = find_players(q, all_players) if q else []
    if q and not hits: st.warning(f'No player found for "{q}". Try just the surname.')
    options = hits or all_players
    default_idx = options.index("Mohamed Salah") if "Mohamed Salah" in options else 0
    player_name = st.selectbox("Select player", options, index=default_idx)

    rows = player_profile[player_profile['Player'] == player_name].sort_values('season')
    if len(rows) > 0:
        latest = rows.iloc[-1]
        st.markdown(f"### {player_name}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Position", latest['Pos']); c2.metric("Latest team", latest['team'])
        c3.metric("Playing style", latest['cluster_name'] if pd.notna(latest['cluster_name']) else "n/a (low minutes)")
        mv = latest['avg_market_value']
        c4.metric("Est. market value", f"€{mv:,.0f}" if pd.notna(mv) and mv > 0 else "not available")

        st.dataframe(player_by_team[player_by_team['Player'] == player_name].sort_values('season')
                     [['season','team','matches_played','total_minutes','total_goals','total_assists']]
                     .rename(columns={'season':'Season','team':'Team','matches_played':'Matches','total_minutes':'Minutes',
                                      'total_goals':'Goals','total_assists':'Assists'}).set_index('Season'))
        col1, col2 = st.columns(2)
        with col1:
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=rows['season'], y=rows['total_goals'], mode='lines+markers', name='Goals', line_color='#2e7d32'))
            fig1.add_trace(go.Scatter(x=rows['season'], y=rows['total_assists'], mode='lines+markers', name='Assists', line_color='#1976d2'))
            fig1.update_layout(title="Goals & assists by season", height=380)
            st.plotly_chart(fig1)
        with col2:
            bars = [c for c in p90_cols if c != 'Touches_p90']
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(y=bars, x=latest[bars].astype(float).values, name=player_name, orientation='h', marker_color='#c62828'))
            fig2.add_trace(go.Bar(y=bars, x=league_avg[bars].values, name='League avg', orientation='h', marker_color='#9e9e9e'))
            fig2.update_layout(title=f"Per-90 vs league average ({latest['season']})", barmode='group', height=380)
            st.plotly_chart(fig2)
            st.caption(f"Touches per 90: {latest['Touches_p90']:.0f} (league average {league_avg['Touches_p90']:.0f}). "
                       "2023-24 data covers only part of the season.")
    else:
        st.warning("No data for this player.")

# ---------------- TAB 3: SENTIMENT TRACKER ----------------
with tab3:
    st.subheader("Fan sentiment assistant")
    st.caption("Try: **goal** · **who scored** · **score** · **sentiment** · **busiest moment** · **accuracy** · **help**  "
               "- or type any fan tweet and the trained model reads its mood.")
    if "msgs" not in st.session_state:
        st.session_state.msgs = [{"role": "assistant", "fig": None,
                                  "text": "Hi! Ask me about Qatar vs Ecuador, or type a fan tweet. Type **help** for ideas."}]
        st.session_state.pending = None
    box = st.container()
    if user := st.chat_input("Type here...", key="fan_chat"):
        st.session_state.msgs.append({"role": "user", "text": user, "fig": None})
        try:
            reply, figk, st.session_state.pending = bot_reply(user, st.session_state.pending)
        except Exception:
            reply, figk = "Sorry, I hit a problem with that one. Try **help**.", None
        st.session_state.msgs.append({"role": "assistant", "text": reply, "fig": figk})
    with box:
        for m in st.session_state.msgs:
            with st.chat_message(m["role"]):
                st.markdown(m["text"])
                if m["fig"]: st.plotly_chart(timeline_fig(zoom=(m["fig"] == "zoom")), key=f"c{id(m)}")

    st.markdown("---")
    st.subheader("Match-timeline sentiment: " + MATCH)
    st.caption("Enner Valencia scored at 16' (penalty) and 31' (header), marked on the chart.")
    st.plotly_chart(timeline_fig(), key="main_timeline")
    st.caption("Classifier: TF-IDF + Logistic Regression trained on labelled World Cup tweets. "
               "Held-out test accuracy: 71.7% (target was 80%; see report for discussion).")
