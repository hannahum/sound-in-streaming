import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Page + theme 
st.set_page_config(
    page_title="Sound in the Streaming Era",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    h1, h2, h3 {
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


sns.set_theme(style="whitegrid")

# Data loading / precomputation
@st.cache_data
def load_data():
    df = pd.read_csv("songs_normalize.csv")
    df["primary_genre"] = df["genre"].apply(lambda x: x.split(",")[0].strip())
    df = df[df["primary_genre"] != "set()"]
    return df


df = load_data()

numeric_features = [
    "danceability",
    "energy",
    "valence",
    "loudness",
    "tempo",
    "acousticness",
    "liveness",
    "speechiness",
    "instrumentalness",
    "duration_ms",
]

yearly = df.groupby("year")[numeric_features].mean().reset_index()
year_genre = (
    df.groupby(["year", "primary_genre"])[numeric_features].mean().reset_index()
)


@st.cache_data
def compute_correlations():
    return df[numeric_features + ["popularity"]].corr()


@st.cache_data
def compute_r2_by_era(start_year, end_year):
    era = df[(df["year"] >= start_year) & (df["year"] <= end_year)]
    X = era[numeric_features]
    y = era["popularity"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LinearRegression().fit(X_train, y_train)
    preds = model.predict(X_test)
    return r2_score(y_test, preds)


corr = compute_correlations()
r2_2000s = compute_r2_by_era(2000, 2009)
r2_2010s = compute_r2_by_era(2010, 2019)

# PAGE RENDER FUNCTIONS

def render_home():
    st.title("Sound in the Streaming Era: What Spotify Data Reveals About Musical Popularity")
    st.image("images/spotify_logo.jpg", use_container_width=True)
    st.subheader(
        """
        This website showcases an analysis of the sonic features of popular songs on Spotify from 2000-2019, providing an interactive exploration of popular music in the streaming era using Spotify data.
        """
    )

    st.markdown("---")


def render_background():
    st.header("Background")
    st.image("images/streaming.webp", width=700)

    st.subheader("The Streaming Era")
    st.markdown(
        """
        In the late 1990s and early 2000s, digital files and the internet completely transformed how we listen to music. Services like Napster (1999) made MP3s easier to
        to share, reflecting that listeners wanted instand access to their favorite tracks rather than physical CDs. 
        In 2003, Apple's iTunes Store was launched, which translated that demand into a legal download store. Devices like the iPod and later the iPhone became portable
        digital libraries that people could use in their everyday lives.

        By the mid 2000s, internet radio and video platforms like Pandora and Youtube, both launched in 2005, introduced streaming music without owning a file.
        This led to the emergence of full streaming services like Spotify, which launched in Europe in 2008 and reached the U.S. in 2011.

        By the years 2014-2016, streaming revenues skyrocketed and digital services surpassed CDs and even digital downloads in global music revenue, making the streaming era as
        the dominant way people hear recorded music.
        """
    )

    img1, img2, img3 = st.columns([3, 2.8, 4.0])

    with img1:
        st.image("images/cds.webp", width=300)
    with img2:
        st.image("images/mp3.webp", width=300)
    with img3:
        st.image("images/ipod.png", width=300)

    timeline_events = [
        ("1999", "Napster launches, making MP3 file-sharing mainstream and "
                "forcing the industry to reckon with digital access."),
        ("2001–2003", "Apple releases the iPod (2001) and iTunes Music Store (2003), "
                    "bringing legal digital downloads to a mass audience."),
        ("2005", "Pandora and YouTube launch, popularizing personalized internet radio "
                "and music video streaming as discovery tools."),
        ("2007–2008", "The iPhone (2007) and app ecosystem, plus Spotify’s launch in "
                    "Europe (2008), make on-demand mobile streaming realistic."),
        ("2011", "Spotify launches in the U.S., pushing the subscription streaming model "
                "into the biggest music market."),
        ("2015", "Apple Music launches; global digital and streaming revenues begin to "
                "overtake CDs and downloads as the industry’s main driver."),
        ("Late 2010s", "Streaming becomes dominant worldwide. Curated playlists, "
                    "algorithms, and platform demographics strongly shape which "
                    "songs rise in Spotify popularity.")
    ]

    for year, description in timeline_events:
        with st.container():
            cols = st.columns([1, 8])
            with cols[0]:
                st.markdown(f"**{year}**")
            with cols[1]:
                st.markdown(description)

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Research Questions")
        st.markdown(
            """
            - Which musical characteristics (danceability, energy, valence, etc.) are
              most associated with Spotify popularity?  
            - How have these features changed across two decades of digital music (2000–2019)?  
            - How do trends vary across genres such as pop, hip hop, rock,
              Dance/Electronic, and Latin?
            """
        )

    with col2:
        st.subheader("Significance")
        st.markdown(
            """
            Streaming platforms and algorithms now shape how we encounter music:
            through **playlists, recommendations, and mood categories**.
            Instead of taking “popular music” for granted, this project asks:

            - Are we choosing what we listen to, or are platforms choosing for us?  
            - Has the sound of popular music adjusted to fit **platform logics** like
              skippability, replayability, and background listening?
            """
        )
    
    with col3:
        st.subheader("What This Project Does")

        st.markdown(
            """
            On the rest of the site, you can:

            - Explore **correlations** between audio features and Spotify popularity  
            - Track **feature trends** over time (danceability, energy, valence,
            loudness, duration)  
            - Compare how those features change **by genre**  
            - See **regression results** that test whether a simple “hit formula”
            exists in the audio data
            """
        )


def render_data_methods():
    st.header("Data & Methods")
    st.image("images/data.jpg", width=300)

    st.subheader("Dataset")
    url = "https://www.kaggle.com/datasets/paradisejoy/top-hits-spotify-from-20002019?resource=download"

    st.markdown(
        f"""
    - **Source:** Kaggle – [Top Hits Spotify from 2000–2019]({url})
    - **Size:** 2,000 tracks (a sample of highly popular songs)  
    - **Audio features:** `artist`, `song`, `duration_ms`, `explicit`, `year`, `popularity`, `danceability`, `energy`,
    `key`, `loudness`, `mode`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`, `genre`
    - **Note:** The popularity score reflects **current Spotify listening**, not necessarily
    how big the song was at release.
    """
    )

    st.markdown("### Sample of the data")
    st.dataframe(df[["artist", "song", "year", "primary_genre", "popularity"]].head(10))
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.image("images/britneyspears.png", width=300)
    with col2:
        st.image("images/destinyschild.jpg", width=300)
    with col3:
        st.image("images/blink182.jpg", width=300)
    with col4:
        st.image("images/nsync.png", width=300)



    st.subheader("Methods")
    
    hero_col1, hero_col2 = st.columns([1.0, 2.7])

    with hero_col1:
        st.image("images/calculations.jpg", width=300)
    with hero_col2:
        st.markdown(
            """
            - Descriptive statistics and feature trends over time (2000–2019)  
            - Genre-level comparisons of key audio features  
            - Correlation analysis between features and popularity  
            - Linear regression models to test whether audio features can
            **predict popularity**  
            - Tools: Python (`pandas`, `seaborn`, `matplotlib`, `scikit-learn`)
            """
        )

def render_sonic_features():
    st.header("Sonic Features & Popularity")

    st.markdown(
        """
        First, I examined correlations between each audio feature and Spotify
        popularity. If there were a simple “hit formula,” we’d expect some features
        to stand out with strong relationships.
        """
    )

    col1, col2 = st.columns([1, 1.2])

    # Correlations with popularity
    with col1:
        st.subheader("Which Features Correlate with Popularity?")
        popularity_corr = corr["popularity"].drop("popularity").sort_values()
        fig, ax = plt.subplots(figsize=(4, 4))
        popularity_corr.plot(kind="barh", ax=ax)
        ax.set_xlabel("Correlation with popularity")
        ax.axvline(0, color="black", linewidth=0.8)
        st.pyplot(fig)

    # Full correlation matrix
    with col2:
        st.subheader("Correlation Matrix (Audio Features + Popularity)")
        fig, ax = plt.subplots(figsize=(5.5, 4))
        sns.heatmap(
            corr,
            cmap="coolwarm",
            center=0,
            annot=False,
            square=True,
            cbar_kws={"shrink": 0.7},
            ax=ax,
        )
        st.pyplot(fig)
    
    st.markdown(
        """
        The correlations with popularity are very small, with nothing above +-0.05. The top positive correlations are duration_ms, loudness, acousticness, speechiness, and tempo. The largest negative correlation was instramentalness. This means that there is no single sonic feature that strongly predicts a hit on Spotify. The strongest signal in the data is actually 
        what is missing from popular tracks: songs with high instramentalness tend to perform worse while tracks with more vocals, volume, and longer duration tend to perform better. However, these differences are only subtle. Therefore, popularity on Spotify isn't driven by a single sound formula, instead reflecting a combination of subtle stylistic choices.
        """
    )

    st.subheader("Key Takeaways:")
    st.markdown(
        """
        - All correlations with popularity are **very small** (close to zero).  
        - Slight positive relationships: **duration**, **loudness**, **acousticness**,
          **speechiness**, **tempo**.  
        - Only clear negative relationship: **instrumentalness**
          (instrumental tracks tend to be a bit less popular).  
        - No individual feature strongly determines popularity – there is
          **no simple acoustic recipe** for a hit in this dataset.
        """
    )

    

    st.header("Can Audio Features Predict Spotify Popularity?")

    st.markdown(
        """
        To test for a measurable “hit formula,” I fit linear regression models that
        predict Spotify popularity from nine audio features, run separately for:
        **2000–2009** and **2010–2019**.
        """
    )

    c1, c2 = st.columns(2)
    with c1:
        st.metric("R² (2000–2009)", f"{r2_2000s:.3f}")
    with c2:
        st.metric("R² (2010–2019)", f"{r2_2010s:.3f}")

    st.markdown(
        """
        **Interpretation of the numbers:**

        - Both R² values are **very low**, meaning the models explain only a tiny fraction
          of the variation in popularity.  
        - The regression itself is working; the low R² tells us that **popularity is not
          strongly determined by these audio features**.  
        - Even though songs are clearly becoming more danceable, energetic, loud, and short,
          those characteristics do **not** reliably predict which songs are more popular
          on Spotify.
        """
    )

def render_genre_shift():
    st.header("Genre Shifts: Do All Styles Change the Same Way?")

    st.markdown(
        """
        The evolution of popular music is not a single straight line. Different genres
        adapt in their own ways while gradually sharing some of the same streaming-era
        strategies: high energy, strong groove, shorter length.
        """
    )

    top_genres = ["hip hop", "Dance/Electronic", "pop", "rock", "latin"]
    subset = year_genre[year_genre["primary_genre"].isin(top_genres)]

    feature = st.selectbox(
        "Choose a feature to view by genre:",
        ["danceability", "energy", "valence", "loudness", "duration_ms"],
        format_func=lambda x: {
            "danceability": "Danceability",
            "energy": "Energy",
            "valence": "Valence (Positivity)",
            "loudness": "Loudness (dB)",
            "duration_ms": "Duration (ms)",
        }[x],
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.lineplot(
        data=subset,
        x="year",
        y=feature,
        hue="primary_genre",
        marker="o",
        ax=ax,
    )
    ax.set_xlabel("Year")
    ax.set_ylabel(feature)
    ax.grid(alpha=0.3)
    ax.legend(title="Genre")
    st.pyplot(fig)

    st.markdown(
        """
        - **Hip hop** becomes steadily more danceable and high-energy, reflecting the rise
          of trap, club-oriented rap, and social media dance trends.  
        - **Dance/Electronic** starts out as a trendsetter in danceability and energy;
          its aesthetics bleed into pop and hip hop over time.  
        - **Pop** looks “middle of the road” numerically, but that stability masks how it
          constantly absorbs elements from hip hop, EDM, and Latin music.  
        - **Rock** stays the least danceable but becomes more groove-aware in the 2010s.  
        - **Latin** shows a strong late-2010s push in danceability and energy, lining up
          with global reggaeton and bilingual pop crossover.
        """
    )

def render_feature_trend():
    st.header("Feature Trends over 2000–2019 (All Genres)")

    st.markdown(
        """
        Here we track how the *average* sound of popular music changed between
        2000–2019 across all genres.
        """
    )

    c1, c2 = st.columns(2)

    # Danceability
    with c1:
        st.subheader("Danceability")
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.lineplot(data=yearly, x="year", y="danceability", marker="o", ax=ax)
        ax.set_ylabel("Danceability (0–1)")
        ax.set_xlabel("Year")
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        st.caption(
            "Danceability dips in the mid-2000s, then rises sharply in the mid-2010s, "
            "aligning with playlist and short-form video cultures that favor rhythm-forward tracks."
        )

    # Energy
    with c2:
        st.subheader("Energy")
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.lineplot(data=yearly, x="year", y="energy", marker="o", ax=ax)
        ax.set_ylabel("Energy (0–1)")
        ax.set_xlabel("Year")
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        st.caption(
            "Energy jumps early in the period and stays high – loud, intense production "
            "becomes a cross-genre norm."
        )

    c3, c4 = st.columns(2)

    # Loudness
    with c3:
        st.subheader("Loudness")
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.lineplot(data=yearly, x="year", y="loudness", marker="o", ax=ax)
        ax.set_ylabel("Loudness (dB)")
        ax.set_xlabel("Year")
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        st.caption(
            "Loudness rises quickly in the early 2000s (the 'loudness wars') and then stabilizes at a high level."
        )

    # Valence
    with c4:
        st.subheader("Valence (Emotional Positivity)")
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.lineplot(data=yearly, x="year", y="valence", marker="o", ax=ax)
        ax.set_ylabel("Valence (0–1)")
        ax.set_xlabel("Year")
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        st.caption(
            "Valence trends downward – popular music remains energetic and loud but "
            "becomes emotionally darker on average."
        )

    st.subheader("Song Duration")
    fig, ax = plt.subplots(figsize=(7, 3))
    sns.lineplot(data=yearly, x="year", y="duration_ms", marker="o", ax=ax)
    ax.set_ylabel("Duration (ms)")
    ax.set_xlabel("Year")
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    st.caption(
        "Average track length decreases over time, consistent with streaming logics that "
        "reward quicker hooks, fewer skips, and more replays."
    )

    st.subheader("Key Takeaways:")
    st.markdown(
        """
        - Overtime, songs have become more high-impact, built for immediate engagement.
        - Popular music in the streaming era is emotionally darker but sonically more aggressive.
        - Shorter songs have more replays, fewer skips, and higher playlist retention.
        - Music adapts structurally to the platform economy, not just listener taste
        """
    )

def render_conclusion():
    st.subheader("Big Picture Takeaway")
    st.markdown(
        """
        Rather than revealing a simple sonic recipe, the regression supports a different story:

        > Popularity in the streaming era is shaped more by **platform dynamics** – playlist
        > placement, recommendation algorithms, label marketing, and social media virality –
        > than by any fixed combination of audio features.
        """
    )

    st.header("Limitations")

    hero_col1, hero_col2 = st.columns([0.5, 1.8])

    with hero_col1:
        st.image("images/pause.jpg", width=260)

    with hero_col2:

        st.markdown(
            """
            - Dataset covers **top-performing songs only** (2,000 tracks), not the full catalog.  
            - Popularity is a **current Spotify metric**, not a historical chart measure.  
            - No data on playlist placement, label promotion, viral trends, or fanbase size,
            all of which clearly affect streams.  
            - Audio features describe the **sound** but not lyrics, visual media, cultural
            context, or nostalgia.  
            - Genre labels are coarse and sometimes messy for hybrid tracks.
            """
        )

    st.header("Next Steps")
    hero_col3, hero_col4 = st.columns([0.5, 1.8])

    with hero_col3:
        st.image("images/skip.webp", width=260)


    with hero_col4:
        st.markdown(
            """
            - Add **lyric sentiment and topic modeling** to connect textual themes with
            musical sound and popularity.  
            - Compare popular tracks with a set of **less-popular / random songs** to see how
            hits differ from the broader catalog.  
            - Incorporate **playlist metadata** to test how exposure and placement influence
            popularity more directly.  
            - Zoom in on a particular **genre or artist** for a more focused case study of
            sonic evolution.  
            - Link Spotify streaming data with **TikTok / short-form video usage** to study
            how audio circulates across platforms.
            """
        )

def render_about():    
    st.header("About This Project")
    st.markdown(
        """
        This project analyzes Spotify audio features (2000–2019) to understand how the
        **sound of popular music** has evolved during the streaming era. The dataset
        includes 100k+ tracks with variables such as danceability, energy, valence,
        tempo, loudness, and more. The goal is to explore how sonic characteristics
        changed over time and how weakly (or strongly) they relate to **popularity**
        in the streaming landscape.
        """
    )

    st.header("Sources")

    st.subheader("History of the Digital & Streaming Era")
    st.markdown(
        """
        - Szalai, Georg — *Napster 20 Years Later: How the File-Sharing Service Changed the Music Industry* (The Hollywood Reporter)  
        - Apple Press Release — *Apple Launches the iTunes Music Store*, 2003  
        - Pandora Media — Company History Overview  
        - Burgess & Green — *YouTube: Online Video and Participatory Culture*  
        - Spotify Newsroom — *A Timeline: Spotify Through the Years*  
        - IFPI — Global Music Reports (2016, 2017, 2019)  
        - Prey, Robert — *Nothing Personal: Algorithmic Individuation on Music Streaming Platforms* (2018)
        """
    )
    st.subheader("Data, Audio Features & Research on Streaming")
    st.markdown(
        """
        - Spotify for Developers — Web API documentation (Audio Features & Popularity)  
        - Schedl, Markus — *The LFM-1b Dataset for Evaluation of Recommender Systems*  
        - Morris & Powers — *Control, Curation and Musical Experience in Spotify*  
        - Eriksson et al. — *Spotify Teardown: Inside the Black Box of Streaming Music* (MIT Press)  
        """
    )
    st.markdown("---")

    st.header("Author")
    st.image("images/author.JPG", width=300)
    st.markdown("Hannah Um")
    st.caption("Undergraduate student pursuing a major in Statistics & Data Science and a minor in Digital Humanities at the University of California, Los Angeles.")
    st.markdown(
        """
        - GitHub: https://github.com/hannahum 
        - LinkedIn: http://www.linkedin.com/in/HannahUm
        """
    )

# TAB NAVIGATION

PAGES = {
    "Home": render_home,
    "Background": render_background,
    "Data & Methods": render_data_methods,
    "Sonic Features & Popularity Analysis": render_sonic_features,
    "Genre Shift Analysis": render_genre_shift,
    "Feature Trend Analysis": render_feature_trend,
    "Conclusion": render_conclusion,
    "About": render_about,
}

tab_names = list(PAGES.keys())
tabs = st.tabs(tab_names)

for tab, name in zip(tabs, tab_names):
    with tab:
        PAGES[name]()
