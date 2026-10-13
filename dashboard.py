"""Receipt dashboard.  Run:  streamlit run dashboard.py

Modes
- No APP_PASSWORD set (your laptop): full access, data from data/output.
- APP_PASSWORD set (online): visitors see a read-only demo built from samples/output;
  entering the password unlocks upload + extraction on data/output.
"""
import hmac
import os
from pathlib import Path

import altair as alt
import streamlit as st
from dotenv import load_dotenv

from src.report import daily_spend, load_results, needs_review, to_excel_bytes, top_items, top_merchants

load_dotenv()
INPUT_DIR, OUTPUT_DIR = Path("data/receipts"), Path("data/output")
DEMO_DIR = Path("samples/output")
INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
APP_PASSWORD = os.getenv("APP_PASSWORD", "")
MAX_UPLOAD = int(os.getenv("MAX_UPLOAD", "10"))  # protects the free API quota

st.set_page_config(page_title="Receipt Dashboard", page_icon="🧾", layout="wide")


def rp(x) -> str:
    return "-" if x is None or x != x else "Rp " + f"{x:,.0f}".replace(",", ".")


BAR_COLOR = "#2563eb"


def show(chart):
    """Full-width chart on both new (width=) and older (use_container_width=) Streamlit."""
    try:
        st.altair_chart(chart, width="stretch")
    except TypeError:
        st.altair_chart(chart, use_container_width=True)


def daily_chart(df):
    """One bar per day that has receipts; dates shown as labels, in order."""
    d = df.assign(label=df["date"].map(lambda x: x.strftime("%d %b")), total_rp=df["total"].map(rp))
    return (alt.Chart(d).mark_bar(color=BAR_COLOR, cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(x=alt.X("label:N", sort=None, title=None, axis=alt.Axis(labelAngle=0)),
                    y=alt.Y("total:Q", title="Rupiah", axis=alt.Axis(format="~s")),
                    tooltip=[alt.Tooltip("label:N", title="Date"), alt.Tooltip("total_rp:N", title="Total")])
            .properties(height=320))


def merchant_chart(df):
    """Horizontal bars, biggest merchant on top, full names visible."""
    d = df.assign(total_rp=df["total"].map(rp), merchant=df["merchant_name"].fillna("(unknown)"))
    return (alt.Chart(d).mark_bar(color=BAR_COLOR, cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
            .encode(y=alt.Y("merchant:N", sort="-x", title=None, axis=alt.Axis(labelLimit=260)),
                    x=alt.X("total:Q", title="Rupiah", axis=alt.Axis(format="~s")),
                    tooltip=[alt.Tooltip("merchant:N", title="Merchant"),
                             alt.Tooltip("receipts:Q", title="Receipts"),
                             alt.Tooltip("total_rp:N", title="Total")])
            .properties(height=320))


# ---------------- Access ----------------
if "unlocked" not in st.session_state:
    st.session_state.unlocked = not APP_PASSWORD  # no password configured = local use, full access


def check_password() -> None:
    entered = st.session_state.get("pw", "")
    st.session_state.unlocked = bool(APP_PASSWORD) and hmac.compare_digest(entered, APP_PASSWORD)
    if not st.session_state.unlocked:
        st.session_state.pw_error = True


demo = not st.session_state.unlocked

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("🧾 Add receipts")
    if demo:
        st.caption("Upload is locked in the public demo.")
        st.text_input("Password", type="password", key="pw", on_change=check_password)
        if st.session_state.pop("pw_error", False):
            st.error("Wrong password.")
    else:
        uploads = st.file_uploader("Receipt photos", type=["jpg", "jpeg", "png", "webp"],
                                   accept_multiple_files=True)
        too_many = bool(uploads) and len(uploads) > MAX_UPLOAD
        if too_many:
            st.warning(f"Max {MAX_UPLOAD} photos per run.")
        if st.button("Extract", type="primary", disabled=not uploads or too_many):
            for f in uploads:
                (INPUT_DIR / Path(f.name).name).write_bytes(f.getbuffer())
            with st.spinner(f"Reading {len(uploads)} receipt(s) with AI..."):
                from src.pipeline import run
                try:
                    stats = run(INPUT_DIR, OUTPUT_DIR, use_db=False, limit=None)
                except Exception as exc:  # setup problems (missing key, bad model) -> readable message
                    st.error(f"Extraction could not start: {exc}")
                    st.stop()
            st.success(f"Processed {stats['processed']} (second opinion on {stats['retried']}), "
                       f"needs review {stats['flagged']}, skipped {stats['skipped']}, failed {stats['failed']}")
            if stats["failed"]:
                st.warning("Some receipts failed. Check the logs for the error.")
            st.rerun()
        st.caption("Photos are saved to data/receipts and results to data/output.")
        if APP_PASSWORD and st.button("Lock"):
            st.session_state.unlocked = False
            st.rerun()

# ---------------- Data ----------------
rec, itm = load_results(DEMO_DIR if demo else OUTPUT_DIR)
st.title("Receipt Dashboard")
if demo:
    st.info("Demo: these are 15 synthetic receipts read by the AI pipeline. "
            "[Source code on GitHub](https://github.com/Kacangpolo/ai-receipt-extractor)")

if rec.empty:
    st.info("No receipts yet. Upload photos in the sidebar, or run `python -m src.pipeline --no-db`.")
    st.stop()

review = needs_review(rec)

# ---------------- KPIs ----------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Receipts", len(rec))
c2.metric("Total spend", rp(rec["total"].sum()))
c3.metric("Avg per receipt", rp(rec["total"].mean()))
c4.metric("Needs review", len(review))

# ---------------- Charts ----------------
left, right = st.columns(2)
with left:
    st.subheader("Spend per day")
    show(daily_chart(daily_spend(rec)))
with right:
    st.subheader("Top merchants")
    show(merchant_chart(top_merchants(rec)))

st.subheader("Top items by spend")
ti = top_items(itm)
st.dataframe(ti.assign(spend=ti["spend"].map(rp)), hide_index=True)

# ---------------- Review queue ----------------
st.subheader("⚠️ Needs review")
if review.empty:
    st.success("All receipts passed the automatic checks.")
else:
    st.dataframe(review.assign(total=review["total"].map(rp)), hide_index=True)

# ---------------- Detail + export ----------------
with st.expander("All receipts"):
    st.dataframe(rec, hide_index=True)
with st.expander("All items"):
    st.dataframe(itm, hide_index=True)

st.download_button("⬇️ Download Excel", data=to_excel_bytes(rec, itm), file_name="receipts.xlsx",
                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
