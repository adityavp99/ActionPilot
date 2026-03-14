import json
import os
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

try:
    from google import genai
except Exception:
    genai = None

st.set_page_config(
    page_title="ActionPilot",
    page_icon="✅",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_NAME = "ActionPilot"
MAX_DAILY_REQUESTS = 20
SAMPLE_DIR = Path("sample_transcripts")

CUSTOM_CSS = """
<style>
:root {
    --bg: #04101d;
    --bg2: #08182b;
    --bg3: #0e2741;
    --stroke: rgba(168, 184, 255, 0.18);
    --stroke-soft: rgba(255,255,255,0.08);
    --text: #f7fbff;
    --muted: #c7d4ff;
    --soft: #8da4d4;
    --purple: #7f6dff;
    --violet: #5346d8;
    --blue: #14345f;
    --cyan: #59a9ff;
    --amber: #f2b56b;
    --panel: rgba(9, 20, 37, 0.72);
}
html, body, [data-testid="stAppViewContainer"] {
    background:
        radial-gradient(circle at 14% 8%, rgba(127,109,255,0.22), transparent 24%),
        radial-gradient(circle at 88% 10%, rgba(89,169,255,0.16), transparent 22%),
        radial-gradient(circle at 52% 42%, rgba(83,70,216,0.12), transparent 28%),
        linear-gradient(180deg, #030713 0%, #071222 26%, #07192d 58%, #081b33 100%);
    color: var(--text);
    font-family: "SF Pro Display", "Avenir Next", "Segoe UI", "Helvetica Neue", sans-serif;
}
[data-testid="stSidebar"] {
    background:
        linear-gradient(180deg, rgba(27, 18, 64, 0.97) 0%, rgba(15, 21, 45, 0.97) 40%, rgba(9, 24, 46, 0.99) 100%);
    border-right: 1px solid rgba(255,255,255,0.1);
    box-shadow: inset -1px 0 0 rgba(255,255,255,0.06);
}
[data-testid="stSidebar"]::before {
    content: "";
    position: absolute;
    inset: 0;
    background:
        radial-gradient(circle at top left, rgba(127,109,255,0.2), transparent 22%),
        radial-gradient(circle at bottom right, rgba(89,169,255,0.12), transparent 26%);
    pointer-events: none;
}
[data-testid="stHeader"] {
    background: transparent !important;
    border-bottom: 0 !important;
}
.block-container {
    max-width: 1360px;
    padding-top: 1.25rem;
    padding-bottom: 2.8rem;
}
h1, h2, h3, h4, h5, h6, p, label, div, span {
    color: var(--text);
}
.hero {
    position: relative;
    overflow: hidden;
    background:
        radial-gradient(circle at top left, rgba(127, 109, 255, 0.34), transparent 24%),
        radial-gradient(circle at 84% 16%, rgba(89, 169, 255, 0.18), transparent 28%),
        linear-gradient(120deg, rgba(69, 45, 152, 0.94) 0%, rgba(20, 42, 79, 0.95) 44%, rgba(10, 57, 82, 0.92) 100%);
    border: 1px solid rgba(208, 223, 255, 0.16);
    border-radius: 28px;
    padding: 30px 32px 28px 32px;
    box-shadow: 0 24px 64px rgba(0,0,0,0.28), inset 0 1px 0 rgba(255,255,255,0.12);
    margin-bottom: 1.35rem;
}
.hero::after {
    content: "";
    position: absolute;
    inset: auto -12% -34% 42%;
    height: 220px;
    background: radial-gradient(circle, rgba(255,255,255,0.12) 0%, transparent 62%);
    transform: rotate(-8deg);
    pointer-events: none;
}
.badge {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.42rem 0.82rem;
    border-radius: 999px;
    font-size: 0.77rem;
    letter-spacing: 0.01em;
    background: rgba(255,255,255,0.09);
    border: 1px solid rgba(255,255,255,0.14);
    color: #eff6ff;
    margin-right: 0.48rem;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.08);
}
.section-title {
    font-size: 2.06rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    margin: 0.35rem 0 1.15rem 0;
}
.subsection-title {
    font-size: 0.98rem;
    font-weight: 750;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #d7e8ff;
    margin: 0.35rem 0 0.55rem 0;
}
.panel-label {
    font-size: 1.08rem;
    font-weight: 700;
    color: #e6f0ff;
    margin: 0.15rem 0 0.8rem 0;
}
.content-title {
    font-size: 2.55rem;
    font-weight: 820;
    letter-spacing: -0.03em;
    margin: 0 0 0.95rem 0;
}
.content-subtitle {
    font-size: 1.42rem;
    font-weight: 760;
    margin: 1.35rem 0 0.72rem 0;
}
.entity-title {
    font-size: 2rem;
    font-weight: 760;
    margin: 0.05rem 0 0.42rem 0;
    color: #b7cdfb;
    font-style: italic;
    letter-spacing: 0.01em;
}
.content-date {
    color: var(--muted);
    font-size: 1.02rem;
    font-weight: 600;
    margin-bottom: 1.35rem;
}
.content-divider {
    height: 1px;
    background: linear-gradient(90deg, rgba(127, 109, 255, 0.34), rgba(89, 169, 255, 0.18), rgba(255,255,255,0.04));
    margin: 1.15rem 0 0.9rem 0;
}
.section-divider {
    height: 1px;
    background: linear-gradient(90deg, rgba(127, 109, 255, 0.32), rgba(89, 169, 255, 0.16), rgba(255,255,255,0.04));
    margin: 1.05rem 0 1.05rem 0;
}
.small-muted {
    color: var(--muted);
    font-size: 1rem;
    line-height: 1.7;
}
.soft {
    color: var(--soft);
}
.metric-card {
    background:
        radial-gradient(circle at 18% 14%, rgba(127,109,255,0.24), transparent 34%),
        radial-gradient(circle at 82% 18%, rgba(89,169,255,0.16), transparent 30%),
        linear-gradient(180deg, rgba(20, 33, 62, 0.97) 0%, rgba(8, 18, 36, 0.99) 100%);
    border: 1px solid rgba(168, 184, 255, 0.26);
    border-radius: 20px;
    box-shadow:
        inset 0 1px 0 rgba(255,255,255,0.1),
        inset 0 -20px 30px rgba(5, 14, 28, 0.18),
        0 18px 30px rgba(1, 8, 20, 0.24);
    padding: 18px 18px 24px 18px;
    min-height: 126px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
}
.metric-card:hover {
    transform: translateY(-2px);
    border-color: rgba(199, 207, 255, 0.38);
    box-shadow:
        inset 0 1px 0 rgba(255,255,255,0.12),
        inset 0 -20px 30px rgba(5, 14, 28, 0.16),
        0 22px 34px rgba(1, 8, 20, 0.28);
}
.metric-label {
    color: #edf4ff;
    font-size: 1.04rem;
    font-weight: 700;
    letter-spacing: 0.01em;
    margin-bottom: 0.76rem;
}
.metric-value {
    color: #ffffff;
    font-size: 2.42rem;
    font-weight: 780;
    line-height: 1;
    text-shadow: 0 6px 20px rgba(0,0,0,0.2);
}
.metrics-stack {
    margin-top: 6.8rem;
}
.metric-status {
    margin-top: 1.25rem;
}
.readiness-shell {
    margin: 0.2rem 0 1.1rem 0;
    padding: 1rem 1rem 1.05rem 1rem;
    border-radius: 22px;
    background:
        radial-gradient(circle at top right, rgba(127,109,255,0.16), transparent 26%),
        linear-gradient(135deg, rgba(18, 31, 56, 0.94) 0%, rgba(10, 20, 38, 0.98) 100%);
    border: 1px solid rgba(151, 171, 237, 0.2);
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.08), 0 18px 34px rgba(3, 10, 22, 0.18);
}
.readiness-grid {
    display: grid;
    grid-template-columns: minmax(170px, 1.15fr) repeat(4, minmax(110px, 1fr));
    gap: 0.7rem;
    align-items: stretch;
}
.readiness-score-card,
.readiness-mini-card {
    border-radius: 18px;
    border: 1px solid rgba(159, 178, 241, 0.18);
    background: linear-gradient(180deg, rgba(27, 41, 71, 0.92) 0%, rgba(14, 24, 43, 0.96) 100%);
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.07);
}
.readiness-score-card {
    padding: 0.95rem 1rem;
    display: flex;
    flex-direction: column;
    justify-content: center;
}
.readiness-kicker {
    color: #dfe9ff;
    font-size: 0.84rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 0.45rem;
}
.readiness-score {
    color: #ffffff;
    font-size: 2.5rem;
    font-weight: 800;
    line-height: 1;
    margin-bottom: 0.4rem;
}
.readiness-reco {
    color: var(--muted);
    font-size: 0.95rem;
    line-height: 1.45;
}
.readiness-mini-card {
    padding: 0.8rem 0.85rem;
    display: flex;
    flex-direction: column;
    justify-content: center;
    min-height: 108px;
}
.readiness-mini-label {
    color: #dce7ff;
    font-size: 0.8rem;
    font-weight: 680;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    line-height: 1.3;
    margin-bottom: 0.5rem;
}
.readiness-mini-value {
    color: #ffffff;
    font-size: 1.9rem;
    font-weight: 780;
    line-height: 1;
}
.stButton button,
.stDownloadButton button {
    border-radius: 14px !important;
    min-height: 46px !important;
    border: 1px solid rgba(196, 210, 255, 0.18) !important;
    background:
        linear-gradient(135deg, rgba(255,255,255,0.08) 0%, rgba(255,255,255,0.02) 36%, transparent 36%),
        radial-gradient(circle at top left, rgba(255,255,255,0.12), transparent 34%),
        linear-gradient(180deg, rgba(45, 61, 95, 0.74) 0%, rgba(21, 31, 49, 0.82) 100%) !important;
    color: #ffffff !important;
    font-weight: 650 !important;
    letter-spacing: 0.01em;
    box-shadow:
        inset 0 1px 0 rgba(255,255,255,0.16),
        inset 0 -12px 18px rgba(7, 14, 28, 0.18),
        0 12px 22px rgba(2, 8, 18, 0.18) !important;
    backdrop-filter: blur(14px) saturate(130%) !important;
    -webkit-backdrop-filter: blur(14px) saturate(130%) !important;
    transition: transform 0.14s ease, box-shadow 0.14s ease, border-color 0.14s ease, filter 0.14s ease !important;
}
.stButton button:hover,
.stDownloadButton button:hover {
    border-color: rgba(216, 226, 255, 0.28) !important;
    background:
        linear-gradient(135deg, rgba(255,255,255,0.12) 0%, rgba(255,255,255,0.03) 38%, transparent 38%),
        radial-gradient(circle at top left, rgba(255,255,255,0.15), transparent 34%),
        linear-gradient(180deg, rgba(54, 73, 111, 0.8) 0%, rgba(27, 40, 63, 0.88) 100%) !important;
    color: #ffffff !important;
    transform: translateY(-1px) !important;
    box-shadow:
        inset 0 1px 0 rgba(255,255,255,0.2),
        inset 0 -12px 18px rgba(7, 14, 28, 0.14),
        0 16px 28px rgba(2, 8, 18, 0.24) !important;
    filter: brightness(1.04) !important;
}
.stButton button:active,
.stDownloadButton button:active {
    transform: translateY(1px) scale(0.99) !important;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.04), 0 6px 12px rgba(2, 8, 18, 0.22) !important;
}
.stButton button:focus-visible,
.stDownloadButton button:focus-visible,
.stTabs [data-baseweb="tab-list"] button:focus-visible,
.stTextArea textarea:focus,
.stTextInput input:focus {
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(62,168,255,0.22), 0 0 0 6px rgba(49,208,170,0.12) !important;
}
.stButton button[kind="primary"] {
    border-color: rgba(186, 178, 255, 0.32) !important;
    background:
        linear-gradient(135deg, rgba(255,255,255,0.12) 0%, rgba(255,255,255,0.04) 34%, transparent 34%),
        radial-gradient(circle at top left, rgba(255,255,255,0.12), transparent 34%),
        linear-gradient(135deg, rgba(104, 87, 239, 0.76) 0%, rgba(79, 63, 211, 0.84) 100%) !important;
    color: #f7f5ff !important;
}
.stButton button[kind="primary"]:hover {
    border-color: rgba(214, 207, 255, 0.42) !important;
    background:
        linear-gradient(135deg, rgba(255,255,255,0.14) 0%, rgba(255,255,255,0.05) 34%, transparent 34%),
        radial-gradient(circle at top left, rgba(255,255,255,0.15), transparent 34%),
        linear-gradient(135deg, rgba(112, 96, 244, 0.82) 0%, rgba(89, 72, 226, 0.9) 100%) !important;
}
.stTextArea textarea, .stTextInput input {
    background:
        linear-gradient(180deg, rgba(14, 21, 37, 0.94) 0%, rgba(10, 16, 30, 0.98) 100%) !important;
    color: #eef4ff !important;
    border-radius: 18px !important;
    border: 1px solid rgba(109, 126, 170, 0.58) !important;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.08), 0 16px 36px rgba(6, 12, 25, 0.22) !important;
    font-size: 1rem !important;
    line-height: 1.65 !important;
}
[data-testid="stTextArea"],
[data-testid="stFileUploader"] {
    position: relative;
    padding: 12px;
    border-radius: 24px;
    background:
        linear-gradient(135deg, rgba(95, 84, 226, 0.24) 0%, rgba(41, 79, 158, 0.16) 54%, rgba(14, 26, 53, 0.22) 100%);
    border: 1px solid rgba(150, 167, 255, 0.22);
    box-shadow:
        inset 0 1px 0 rgba(255,255,255,0.09),
        inset 0 -18px 28px rgba(5, 14, 28, 0.18),
        0 20px 48px rgba(2, 9, 20, 0.22);
    backdrop-filter: blur(18px);
}
[data-testid="stTextArea"]::before,
[data-testid="stFileUploader"]::before {
    content: "";
    position: absolute;
    inset: 0;
    border-radius: 24px;
    background:
        radial-gradient(circle at top left, rgba(131, 160, 255, 0.18), transparent 30%),
        linear-gradient(180deg, rgba(255,255,255,0.05), transparent 22%);
    pointer-events: none;
}
[data-testid="stTextArea"] > div,
[data-testid="stFileUploader"] > div {
    position: relative;
    z-index: 1;
}
[data-testid="stTextArea"]:hover,
[data-testid="stFileUploader"]:hover {
    border-color: rgba(182, 196, 255, 0.28);
    box-shadow:
        inset 0 1px 0 rgba(255,255,255,0.1),
        inset 0 -18px 28px rgba(5, 14, 28, 0.14),
        0 24px 54px rgba(2, 9, 20, 0.24);
}
section[data-testid="stFileUploaderDropzone"] {
    background: linear-gradient(180deg, rgba(20, 29, 48, 0.92) 0%, rgba(11, 18, 32, 0.98) 100%) !important;
    border: 1px solid rgba(111, 129, 173, 0.54) !important;
    border-radius: 18px !important;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.08), 0 16px 36px rgba(6, 12, 25, 0.18) !important;
    transition: transform 0.16s ease, box-shadow 0.16s ease, border-color 0.16s ease !important;
}
section[data-testid="stFileUploaderDropzone"]:hover {
    transform: translateY(-1px);
    border-color: rgba(112, 158, 224, 0.72) !important;
    box-shadow: 0 20px 38px rgba(6, 12, 25, 0.18) !important;
}
section[data-testid="stFileUploader"] button,
section[data-testid="stFileUploaderDropzone"] button,
section[data-testid="stFileUploaderDropzone"] button *,
section[data-testid="stFileUploader"] button * {
    color: #e7efff !important;
    font-weight: 600 !important;
}
section[data-testid="stFileUploader"] button,
section[data-testid="stFileUploaderDropzone"] button {
    background: linear-gradient(180deg, rgba(54, 70, 102, 0.96) 0%, rgba(35, 48, 74, 0.98) 100%) !important;
    border: 1px solid rgba(153, 171, 220, 0.34) !important;
    color: #eaf1ff !important;
    transform: none !important;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.08), 0 10px 22px rgba(5, 11, 22, 0.22) !important;
}
section[data-testid="stFileUploader"] button:hover,
section[data-testid="stFileUploaderDropzone"] button:hover {
    background: linear-gradient(135deg, rgba(103, 86, 236, 0.98) 0%, rgba(71, 111, 226, 0.96) 100%) !important;
    border-color: rgba(203, 194, 255, 0.54) !important;
    color: #ffffff !important;
    transform: translateY(-1px) !important;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.12), 0 16px 30px rgba(12, 18, 42, 0.32) !important;
}
section[data-testid="stFileUploaderDropzone"] p,
section[data-testid="stFileUploaderDropzone"] small,
section[data-testid="stFileUploaderDropzone"] span,
section[data-testid="stFileUploaderDropzone"] svg {
    color: #ccd7f0 !important;
}
[data-baseweb="select"] > div {
    background: linear-gradient(180deg, rgba(16, 25, 43, 0.98) 0%, rgba(10, 18, 32, 1) 100%) !important;
    color: #f4f8ff !important;
    border-radius: 14px !important;
    border: 1px solid rgba(138, 157, 212, 0.62) !important;
    box-shadow: 0 10px 24px rgba(6, 12, 25, 0.18) !important;
}
[data-baseweb="select"]:focus-within > div {
    border-color: rgba(181, 196, 255, 0.88) !important;
    box-shadow: 0 0 0 3px rgba(127, 109, 255, 0.16), 0 10px 24px rgba(6, 12, 25, 0.18) !important;
}
[data-baseweb="select"] input {
    color: #f4f8ff !important;
    caret-color: #ffffff !important;
}
[data-baseweb="select"] input::placeholder {
    color: #cdd9f8 !important;
    opacity: 1 !important;
}
[data-testid="stWidgetLabel"] {
    margin-bottom: 0.45rem !important;
}
[data-testid="stWidgetLabel"] p {
    color: #e4efff !important;
    font-size: 1.06rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.01em !important;
}
[data-baseweb="select"] span,
[data-baseweb="select"] div {
    color: #f4f8ff !important;
}
[data-baseweb="select"] svg {
    color: #f4f8ff !important;
    fill: #f4f8ff !important;
    opacity: 1 !important;
}
[data-baseweb="select"] [aria-hidden="true"] {
    color: #f4f8ff !important;
    fill: #f4f8ff !important;
    opacity: 1 !important;
}
[data-baseweb="tag"] {
    background: linear-gradient(180deg, rgba(82, 96, 138, 0.92) 0%, rgba(63, 76, 112, 0.96) 100%) !important;
    color: #ffffff !important;
    border: 1px solid rgba(178, 191, 235, 0.34) !important;
    border-radius: 10px !important;
}
[data-baseweb="tag"] span {
    color: #ffffff !important;
}
[data-baseweb="tag"] svg,
[data-baseweb="tag"] button,
[data-baseweb="tag"] [role="button"] {
    color: #ffffff !important;
    fill: #ffffff !important;
    opacity: 1 !important;
}
[data-baseweb="popover"] {
    background: linear-gradient(180deg, rgba(16, 25, 43, 0.99) 0%, rgba(10, 18, 32, 1) 100%) !important;
    border: 1px solid rgba(138, 157, 212, 0.58) !important;
    box-shadow: 0 18px 34px rgba(6, 12, 25, 0.34) !important;
}
[data-baseweb="popover"] > div,
[data-baseweb="popover"] ul,
[data-baseweb="popover"] li {
    background: transparent !important;
}
ul[role="listbox"] {
    background: linear-gradient(180deg, rgba(16, 25, 43, 0.99) 0%, rgba(10, 18, 32, 1) 100%) !important;
    padding: 0.35rem !important;
}
li[role="option"] {
    background: transparent !important;
    color: #f4f8ff !important;
    border-radius: 10px !important;
    margin: 0.08rem 0 !important;
}
li[role="option"] * {
    color: #f4f8ff !important;
}
li[role="option"]:hover {
    background: rgba(86, 104, 160, 0.42) !important;
    color: #ffffff !important;
}
li[role="option"]:hover * {
    color: #ffffff !important;
}
li[role="option"][aria-selected="true"] {
    background: rgba(107, 126, 188, 0.55) !important;
    color: #ffffff !important;
}
li[role="option"][aria-selected="true"] * {
    color: #ffffff !important;
}
li[role="option"][aria-disabled="false"] {
    opacity: 1 !important;
}
[data-baseweb="select"] [title],
[data-baseweb="popover"] [title] {
    color: #f4f8ff !important;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 0.2rem;
    border-bottom: 1px solid rgba(110, 143, 196, 0.18);
}
.stTabs [data-baseweb="tab-list"] button {
    background: transparent !important;
    border-radius: 999px !important;
    border: 1px solid transparent !important;
    font-size: 1.06rem !important;
    font-weight: 700 !important;
    padding: 0.55rem 0.92rem 0.7rem 0.92rem !important;
    margin-right: 0.3rem !important;
    transition: transform 0.14s ease, background 0.14s ease, border-color 0.14s ease, color 0.14s ease !important;
}
.stTabs [data-baseweb="tab-list"] button p,
.stTabs [data-baseweb="tab-list"] button span {
    font-size: 1.06rem !important;
    font-weight: 700 !important;
}
.stTabs [data-baseweb="tab-list"] button:hover {
    background: rgba(26, 43, 71, 0.62) !important;
    border-color: rgba(125,159,214,0.16) !important;
    transform: translateY(-1px);
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(180deg, rgba(76, 63, 178, 0.3) 0%, rgba(16, 32, 58, 0.94) 100%) !important;
    border-color: rgba(161,177,255,0.28) !important;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.08), 0 8px 18px rgba(7, 13, 30, 0.18);
}
.stDataFrame, .stTable, [data-testid="stDataEditor"] {
    border-radius: 18px !important;
    overflow: hidden;
    border: 1px solid rgba(143, 160, 214, 0.18);
    box-shadow: 0 12px 28px rgba(3, 10, 22, 0.14);
}
[data-testid="stDataEditor"] input,
[data-testid="stDataEditor"] textarea,
[data-testid="stDataEditor"] select,
[data-testid="stDataEditor"] option,
[data-testid="stDataEditor"] optgroup,
[data-testid="stDataEditor"] [contenteditable="true"],
[data-testid="stDataEditor"] [data-baseweb="select"] *,
[data-testid="stDataEditor"] [role="gridcell"],
[data-testid="stDataEditor"] [role="gridcell"] * {
    color: #0f172a !important;
    -webkit-text-fill-color: #0f172a !important;
}
[data-testid="stDataEditor"] select {
    background: #ffffff !important;
}
[data-testid="stDataEditor"] select option,
[data-testid="stDataEditor"] option,
select option {
    color: #0f172a !important;
    background: #ffffff !important;
}
[data-testid="stDataEditor"] select option,
[data-testid="stDataEditor"] option {
    color: #0f172a !important;
    background: #ffffff !important;
}
[data-testid="stDataEditor"] select:focus option:checked,
[data-testid="stDataEditor"] option:checked {
    color: #0f172a !important;
    background: #dbe4f3 !important;
}
[data-testid="stDataEditor"] [data-baseweb="popover"],
[data-testid="stDataEditor"] [data-baseweb="menu"],
[data-testid="stDataEditor"] ul[role="listbox"] {
    background: linear-gradient(180deg, rgba(244, 247, 252, 0.99) 0%, rgba(231, 236, 244, 1) 100%) !important;
    border: 1px solid rgba(191, 199, 214, 0.9) !important;
    box-shadow: 0 14px 28px rgba(6, 12, 25, 0.16) !important;
}
[data-testid="stDataEditor"] li[role="option"] {
    background: transparent !important;
    color: #0f172a !important;
}
[data-testid="stDataEditor"] li[role="option"] * {
    color: #0f172a !important;
}
[data-testid="stDataEditor"] li[role="option"]:hover,
[data-testid="stDataEditor"] li[role="option"][aria-selected="true"] {
    background: rgba(148, 163, 184, 0.22) !important;
    color: #0f172a !important;
}
[data-testid="stDataEditor"] li[role="option"]:hover *,
[data-testid="stDataEditor"] li[role="option"][aria-selected="true"] * {
    color: #0f172a !important;
}
.good-strip {
    background: linear-gradient(135deg, rgba(36, 43, 88, 0.94) 0%, rgba(20, 41, 82, 0.86) 100%);
    border: 1px solid rgba(142, 161, 255, 0.22);
    color: #eef4ff;
    padding: 0.95rem 1rem;
    border-radius: 18px;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.06), 0 10px 24px rgba(3, 12, 18, 0.16);
}
.warn-strip {
    background: linear-gradient(135deg, rgba(73, 53, 94, 0.84) 0%, rgba(52, 42, 84, 0.92) 100%);
    border: 1px solid rgba(184, 162, 255, 0.22);
    color: #fff6dd;
    padding: 0.95rem 1rem;
    border-radius: 18px;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.06), 0 10px 24px rgba(16, 10, 6, 0.12);
}
hr {
    border: none;
    border-top: 1px solid rgba(137, 152, 218, 0.14);
}
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] .stSelectbox {
    position: relative;
    z-index: 1;
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] li,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stCaption {
    color: #d9e6ff !important;
}
[data-testid="stSidebar"] h2 {
    font-size: 1.85rem !important;
    letter-spacing: -0.03em;
    margin-bottom: 0.15rem;
}
[data-testid="stSidebar"] ul {
    padding-left: 1rem;
}
[data-testid="stSidebar"] hr {
    border-top: 1px solid rgba(255,255,255,0.08);
}
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: linear-gradient(180deg, rgba(19, 27, 49, 0.96) 0%, rgba(12, 20, 36, 0.98) 100%) !important;
}
[data-testid="stVerticalBlock"] [data-testid="stTextArea"] {
    margin-bottom: 0.4rem;
}
[data-testid="stFileUploader"] {
    margin-bottom: 1.1rem;
}
@media (max-width: 1100px) {
    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
    }
    .hero {
        padding: 24px 24px 22px 24px;
        border-radius: 24px;
    }
    .hero h1 {
        font-size: 2.5rem !important;
    }
    .metrics-stack {
        margin-top: 2rem;
    }
    .readiness-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
    .readiness-score-card {
        grid-column: 1 / -1;
    }
    .content-title {
        font-size: 2.2rem;
    }
    .entity-title {
        font-size: 1.8rem;
    }
}
@media (max-width: 768px) {
    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
        padding-top: 0.85rem;
    }
    .hero {
        padding: 20px 18px 20px 18px;
        border-radius: 22px;
        margin-bottom: 1rem;
    }
    .hero h1 {
        font-size: 2rem !important;
        line-height: 1.1 !important;
    }
    .badge {
        margin-bottom: 0.45rem;
    }
    .section-title {
        font-size: 1.7rem;
    }
    .content-title {
        font-size: 1.95rem;
    }
    .content-subtitle {
        font-size: 1.2rem;
    }
    .entity-title {
        font-size: 1.55rem;
    }
    .metric-card {
        min-height: 112px;
        padding: 16px 14px 18px 14px;
    }
    .readiness-grid {
        grid-template-columns: 1fr;
    }
    .readiness-mini-card,
    .readiness-score-card {
        min-height: unset;
    }
    .metric-value {
        font-size: 2.05rem;
    }
    .metrics-stack {
        margin-top: 1.1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        overflow-x: auto;
        overflow-y: hidden;
        flex-wrap: nowrap !important;
        padding-bottom: 0.35rem;
        scrollbar-width: thin;
    }
    .stTabs [data-baseweb="tab-list"] button {
        white-space: nowrap;
    }
    .stButton button,
    .stDownloadButton button {
        min-height: 44px !important;
    }
    .stTextArea textarea,
    .stTextInput input {
        font-size: 0.96rem !important;
    }
}
@media (max-width: 520px) {
    .hero h1 {
        font-size: 1.7rem !important;
    }
    .small-muted {
        font-size: 0.92rem;
    }
    .section-title {
        font-size: 1.5rem;
    }
    .subsection-title {
        font-size: 0.9rem;
    }
    .content-title {
        font-size: 1.72rem;
    }
    .content-subtitle {
        font-size: 1.08rem;
    }
    .entity-title {
        font-size: 1.4rem;
    }
    .metric-label {
        font-size: 0.92rem;
    }
    .metric-value {
        font-size: 1.82rem;
    }
    .good-strip,
    .warn-strip {
        padding: 0.85rem 0.85rem;
    }
}
</style>
"""

EXTRACTION_SCHEMA = {
    "meeting_title": "string",
    "meeting_date": "string",
    "summary": "string",
    "decisions": ["string"],
    "risks_blockers": ["string"],
    "action_items": [
        {
            "task": "string",
            "owner": "string",
            "deadline": "string",
            "priority": "High | Medium | Low",
            "status": "Not Started | In Progress | Blocked | Done",
            "notes": "string",
        }
    ],
}

DEFAULT_TRANSCRIPT = """Meeting: Product Launch Sync
Date: 2026-03-10

Sarah: We need the landing page finalized by Friday.
Arjun: I can own the landing page copy and coordinate with design.
Mei: Analytics events are still incomplete. I need support from DevOps.
Ravi: I will complete analytics instrumentation by next Tuesday.
Sarah: Good. We also need legal approval for the pricing page.
Nina: I will send the pricing page to legal by tomorrow 2 PM.
Sarah: Decision taken: launch date remains March 28 unless legal flags a blocker.
Mei: Main risk is incomplete analytics coverage before UAT.
"""


def init_state() -> None:
    today = datetime.now().strftime("%Y-%m-%d")
    defaults = {
        "parsed": None,
        "edited_df": None,
        "transcript_text": DEFAULT_TRANSCRIPT,
        "app_name": APP_NAME,
        "request_date": today,
        "request_count": 0,
        "last_source": "Starter sample",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    if st.session_state.request_date != today:
        st.session_state.request_date = today
        st.session_state.request_count = 0


def get_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets["GEMINI_API_KEY"]
        except Exception:
            api_key = None

    if not api_key or genai is None:
        return None
    return genai.Client(api_key=api_key)


def can_call_model() -> bool:
    return int(st.session_state.request_count) < MAX_DAILY_REQUESTS


def register_call() -> None:
    st.session_state.request_count = int(st.session_state.request_count) + 1


def load_sample_files() -> Dict[str, str]:
    samples = {}
    if SAMPLE_DIR.exists() and SAMPLE_DIR.is_dir():
        for path in sorted(SAMPLE_DIR.glob("*.txt")):
            try:
                samples[path.name] = path.read_text(encoding="utf-8")
            except Exception:
                pass
    return samples


def extract_meeting_data(transcript: str) -> Dict[str, Any]:
    client = get_client()
    if client is None:
        return mock_extract(transcript)

    prompt = f"""
You are an expert chief of staff and meeting operations analyst.
Extract structured information from the meeting transcript.
Return ONLY valid JSON.

Schema:
{json.dumps(EXTRACTION_SCHEMA, indent=2)}

Rules:
- Infer owners only when strongly supported by the transcript.
- Keep tasks atomic and action-oriented.
- Normalize priority conservatively.
- If deadline is not explicit, return an empty string.
- Keep summary under 120 words.
- Avoid hallucinating people or decisions.
- Merge duplicates.
- Do not convert a risk into an action item unless the transcript explicitly assigns work.

Transcript:
{transcript}
"""
    try:
        register_call()
        response = client.models.generate_content(
            model=current_model_name(),
            contents=prompt,
        )
        text = response.text.strip()
        start = text.find("{")
        end = text.rfind("}")
        cleaned = text[start : end + 1] if start != -1 and end != -1 else text
        return json.loads(cleaned)
    except Exception:
        return mock_extract(transcript)


def mock_extract(transcript: str) -> Dict[str, Any]:
    return {
        "meeting_title": "Product Launch Sync",
        "meeting_date": datetime.now().strftime("%Y-%m-%d"),
        "summary": "The team reviewed launch readiness, confirmed the launch date remains in place, and aligned on next actions for the landing page, analytics instrumentation, and legal review.",
        "decisions": ["Launch date remains March 28 unless legal flags a blocker."],
        "risks_blockers": [
            "Analytics coverage may be incomplete before UAT.",
            "Legal approval may affect pricing page readiness.",
        ],
        "action_items": [
            {
                "task": "Finalize landing page copy and coordinate with design.",
                "owner": "Arjun",
                "deadline": "Friday",
                "priority": "High",
                "status": "Not Started",
                "notes": "Critical for launch readiness.",
            },
            {
                "task": "Complete analytics instrumentation.",
                "owner": "Ravi",
                "deadline": "Next Tuesday",
                "priority": "High",
                "status": "In Progress",
                "notes": "Support needed from DevOps.",
            },
            {
                "task": "Send pricing page to legal for approval.",
                "owner": "Nina",
                "deadline": "Tomorrow 2 PM",
                "priority": "Medium",
                "status": "Not Started",
                "notes": "Potential blocker to launch content.",
            },
        ],
    }


def to_dataframe(parsed: Dict[str, Any]) -> pd.DataFrame:
    items = parsed.get("action_items", []) if parsed else []
    cols = ["task", "owner", "deadline", "priority", "status", "notes"]
    if not items:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(items)
    for col in cols:
        if col not in df.columns:
            df[col] = ""
    return df[cols]


def owner_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=["owner", "total_tasks", "high_priority", "open_items"]
        )

    temp = df.copy()
    temp["owner"] = temp["owner"].fillna("").replace("", "Unassigned")
    temp["is_high"] = temp["priority"].eq("High")
    temp["is_open"] = temp["status"].isin(["Not Started", "In Progress", "Blocked"])

    out = (
        temp.groupby("owner", dropna=False)
        .agg(
            total_tasks=("task", "count"),
            high_priority=("is_high", "sum"),
            open_items=("is_open", "sum"),
        )
        .reset_index()
        .sort_values(["open_items", "high_priority", "total_tasks"], ascending=False)
    )
    return out


def analyze_execution_readiness(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {
            "total_tasks": 0,
            "missing_owners": 0,
            "missing_deadlines": 0,
            "high_priority_missing_deadline": 0,
            "high_priority_missing_owner": 0,
            "readiness_score": 100,
            "recommendation": "This action package is ready for execution.",
        }

    temp = df.fillna("").copy()
    temp["owner"] = temp["owner"].astype(str).str.strip()
    temp["deadline"] = temp["deadline"].astype(str).str.strip()
    temp["priority"] = temp["priority"].astype(str).str.strip()

    missing_owner_mask = temp["owner"].eq("")
    missing_deadline_mask = temp["deadline"].eq("")
    high_priority_mask = temp["priority"].eq("High")

    missing_owners = int(missing_owner_mask.sum())
    missing_deadlines = int(missing_deadline_mask.sum())
    high_priority_missing_deadline = int(
        (high_priority_mask & missing_deadline_mask).sum()
    )
    high_priority_missing_owner = int((high_priority_mask & missing_owner_mask).sum())

    score = 100
    score -= 15 * missing_owners
    score -= 10 * missing_deadlines
    score -= 15 * high_priority_missing_deadline
    score -= 10 * high_priority_missing_owner
    readiness_score = max(0, min(100, score))

    if missing_owners > 0 and missing_deadlines > 0:
        recommendation = "Assign owners and due dates before execution."
    elif missing_owners > 0:
        recommendation = "Assign owners before execution."
    elif missing_deadlines > 0:
        recommendation = "Add due dates before execution."
    else:
        recommendation = "This action package is ready for execution."

    return {
        "total_tasks": int(len(temp)),
        "missing_owners": missing_owners,
        "missing_deadlines": missing_deadlines,
        "high_priority_missing_deadline": high_priority_missing_deadline,
        "high_priority_missing_owner": high_priority_missing_owner,
        "readiness_score": readiness_score,
        "recommendation": recommendation,
    }


def merge_filtered_edits(
    base_df: pd.DataFrame, original_filtered_df: pd.DataFrame, edited_filtered_df: pd.DataFrame
) -> pd.DataFrame:
    cols = ["task", "owner", "deadline", "priority", "status", "notes"]

    base = base_df.copy()
    if base.empty:
        return edited_filtered_df.copy()[cols] if not edited_filtered_df.empty else pd.DataFrame(columns=cols)

    base = base.reindex(columns=cols)
    original_filtered = original_filtered_df.copy().reindex(columns=cols)
    edited_filtered = edited_filtered_df.copy().reindex(columns=cols)

    deleted_indices = [idx for idx in original_filtered.index if idx not in edited_filtered.index]
    if deleted_indices:
        base = base.drop(index=[idx for idx in deleted_indices if idx in base.index])

    shared_indices = [idx for idx in edited_filtered.index if idx in base.index]
    if shared_indices:
        base.loc[shared_indices, cols] = edited_filtered.loc[shared_indices, cols]

    new_indices = [idx for idx in edited_filtered.index if idx not in base.index]
    if new_indices:
        additions = edited_filtered.loc[new_indices, cols]
        base = pd.concat([base, additions], axis=0)

    return base[cols]


def current_model_name() -> str:
    return os.getenv("GEMINI_MODEL", "gemini-2.0-flash")


def render_metric_card(label: str, value: int) -> None:
    st.markdown(
        f"""
        <div class='metric-card'>
            <div class='metric-label'>{label}</div>
            <div class='metric-value'>{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_execution_readiness(stats: Dict[str, Any]) -> None:
    st.markdown(
        "<div class='panel-label'>Execution Readiness</div>", unsafe_allow_html=True
    )
    st.markdown(
        f"""
        <div class='readiness-shell'>
            <div class='readiness-grid'>
                <div class='readiness-score-card'>
                    <div class='readiness-kicker'>Readiness Score</div>
                    <div class='readiness-score'>{stats['readiness_score']}</div>
                    <div class='readiness-reco'>{stats['recommendation']}</div>
                </div>
                <div class='readiness-mini-card'>
                    <div class='readiness-mini-label'>Total Tasks</div>
                    <div class='readiness-mini-value'>{stats['total_tasks']}</div>
                </div>
                <div class='readiness-mini-card'>
                    <div class='readiness-mini-label'>Missing Owners</div>
                    <div class='readiness-mini-value'>{stats['missing_owners']}</div>
                </div>
                <div class='readiness-mini-card'>
                    <div class='readiness-mini-label'>Missing Deadlines</div>
                    <div class='readiness-mini-value'>{stats['missing_deadlines']}</div>
                </div>
                <div class='readiness-mini-card'>
                    <div class='readiness-mini-label'>High Priority Gaps</div>
                    <div class='readiness-mini-value'>{stats['high_priority_missing_owner'] + stats['high_priority_missing_deadline']}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def generate_simple_pdf(title: str, lines: List[str]) -> bytes:
    max_lines_per_page = 42
    wrapped_lines: List[str] = []
    for line in lines:
        chunks = textwrap.wrap(line, width=92) if line else [""]
        wrapped_lines.extend(chunks)

    if not wrapped_lines:
        wrapped_lines = [""]

    pages = [
        wrapped_lines[i : i + max_lines_per_page]
        for i in range(0, len(wrapped_lines), max_lines_per_page)
    ]

    objects: list[str] = []
    page_object_numbers: list[int] = []
    current_obj = 4

    for page_lines in pages:
        page_obj = current_obj
        content_obj = current_obj + 1
        current_obj += 2
        page_object_numbers.append(page_obj)

        stream_lines = [
            "BT",
            "/F1 22 Tf",
            "50 788 Td",
            f"({pdf_escape(title)}) Tj",
            "0 -28 Td",
            "/F1 11 Tf",
        ]
        for idx, line in enumerate(page_lines):
            if idx > 0:
                stream_lines.append("0 -16 Td")
            stream_lines.append(f"({pdf_escape(line)}) Tj")
        stream_lines.append("ET")
        stream = "\n".join(stream_lines)

        objects.append(
            f"{page_obj} 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 842] /Resources << /Font << /F1 3 0 R >> >> /Contents {content_obj} 0 R >>\nendobj\n"
        )
        objects.append(
            f"{content_obj} 0 obj\n<< /Length {len(stream.encode('latin-1', errors='replace'))} >>\nstream\n{stream}\nendstream\nendobj\n"
        )

    pages_kids = " ".join(f"{obj_num} 0 R" for obj_num in page_object_numbers)
    header_objects = [
        "1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n",
        f"2 0 obj\n<< /Type /Pages /Kids [{pages_kids}] /Count {len(page_object_numbers)} >>\nendobj\n",
        "3 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n",
    ]

    pdf_parts = ["%PDF-1.4\n"]
    offsets = [0]
    all_objects = header_objects + objects

    for obj in all_objects:
        offsets.append(sum(len(part.encode("latin-1")) for part in pdf_parts))
        pdf_parts.append(obj)

    xref_offset = sum(len(part.encode("latin-1")) for part in pdf_parts)
    pdf_parts.append(f"xref\n0 {len(offsets)}\n")
    pdf_parts.append("0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf_parts.append(f"{offset:010d} 00000 n \n")
    pdf_parts.append(
        f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF"
    )
    return "".join(pdf_parts).encode("latin-1", errors="replace")


def build_action_brief_pdf(parsed: Dict[str, Any], df: pd.DataFrame) -> bytes:
    title = parsed.get("meeting_title", "Meeting Action Brief")
    meeting_date = parsed.get("meeting_date", "")
    lines = [
        f"Meeting: {title}",
        f"Date: {meeting_date}",
        "",
        "Executive Summary",
        parsed.get("summary", ""),
        "",
        "Decisions",
    ]
    decisions = parsed.get("decisions", [])
    if decisions:
        lines.extend(f"- {item}" for item in decisions)
    else:
        lines.append("- None")
    lines.extend(["", "Risks / Blockers"])
    risks = parsed.get("risks_blockers", [])
    if risks:
        lines.extend(f"- {item}" for item in risks)
    else:
        lines.append("- None")
    lines.extend(["", "Action Items"])
    if df.empty:
        lines.append("- No action items extracted.")
    else:
        for _, row in df.fillna("").iterrows():
            owner = row.get("owner", "") or "Unassigned"
            deadline = row.get("deadline", "") or "No deadline"
            priority = row.get("priority", "") or "Unspecified"
            status = row.get("status", "") or "Not Started"
            lines.append(
                f"- {row.get('task', '')} | Owner: {owner} | Due: {deadline} | Priority: {priority} | Status: {status}"
            )
    return generate_simple_pdf(title, lines)


def build_follow_up_email(parsed: Dict[str, Any], df: pd.DataFrame) -> str:
    meeting_title = parsed.get("meeting_title", "Meeting")
    meeting_date = parsed.get("meeting_date", "")
    summary = parsed.get("summary", "")

    email_lines = [
        f"Subject: Follow-up | {meeting_title} | {meeting_date}",
        "",
        "Hi team,",
        "",
        f"Here is the follow-up from {meeting_title} held on {meeting_date}.",
        "",
        "Summary",
        summary,
        "",
        "Decisions",
    ]

    decisions = parsed.get("decisions", [])
    if decisions:
        email_lines.extend(f"- {item}" for item in decisions)
    else:
        email_lines.append("- No formal decisions captured.")

    email_lines.extend(["", "Action Items"])
    if df.empty:
        email_lines.append("- No action items captured.")
    else:
        for _, row in df.fillna("").iterrows():
            owner = row.get("owner", "") or "Unassigned"
            deadline = row.get("deadline", "") or "No deadline"
            status = row.get("status", "") or "Not Started"
            email_lines.append(
                f"- {owner}: {row.get('task', '')} (Due: {deadline}; Status: {status})"
            )

    risks = parsed.get("risks_blockers", [])
    if risks:
        email_lines.extend(["", "Risks / Blockers"])
        email_lines.extend(f"- {item}" for item in risks)

    email_lines.extend(["", "Thanks,"])
    return "\n".join(email_lines)


def build_task_tracker_csv(parsed: Dict[str, Any], df: pd.DataFrame) -> bytes:
    tracker_df = df.fillna("").copy()
    rename_map = {
        "task": "Task Name",
        "owner": "Assignee",
        "deadline": "Due Date",
        "priority": "Priority",
        "status": "Status",
        "notes": "Notes",
    }
    tracker_df = tracker_df.rename(columns=rename_map)
    tracker_df["Meeting"] = parsed.get("meeting_title", "")
    tracker_df["Meeting Date"] = parsed.get("meeting_date", "")
    tracker_df["Type"] = "Action Item"
    tracker_df["Created From"] = "ActionPilot"
    ordered_cols = [
        "Task Name",
        "Assignee",
        "Due Date",
        "Status",
        "Priority",
        "Notes",
        "Meeting",
        "Meeting Date",
        "Type",
        "Created From",
    ]
    for col in ordered_cols:
        if col not in tracker_df.columns:
            tracker_df[col] = ""
    return tracker_df[ordered_cols].to_csv(index=False).encode("utf-8")


def render_header() -> None:
    st.markdown(
        """
        <div class='hero'>
            <div>
                <span class='badge'>AI Meeting Intelligence</span>
                <span class='badge'>Action Tracking</span>
                <span class='badge'>Ops Workflow Ready</span>
            </div>
            <h1 style='margin:0.9rem 0 0.45rem 0; font-size:3rem; line-height:1.05;'>Turn messy meeting notes into accountable action.</h1>
            <p class='small-muted' style='max-width: 900px; margin-bottom:0;'>Upload or paste a meeting transcript, extract decisions and tasks, review them in a professional workspace, and export the action register for execution.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(samples: Dict[str, str]) -> None:
    with st.sidebar:
        st.markdown(f"## {APP_NAME}")
        st.caption("AI meeting summary + action tracker")
        st.markdown("---")
        st.write("**Workspace**")
        st.markdown(
            "- Paste or upload transcript\n"
            "- Extract meeting intelligence\n"
            "- Review and edit action register\n"
            "- Export CSV or JSON"
        )
        st.markdown("---")
        st.write("**Usage guardrail**")
        left = max(0, MAX_DAILY_REQUESTS - int(st.session_state.request_count))
        st.markdown(
            f"<div class='warn-strip'>Daily AI requests used: <b>{st.session_state.request_count}</b> / {MAX_DAILY_REQUESTS}<br>Remaining today: <b>{left}</b></div>",
            unsafe_allow_html=True,
        )
        st.markdown("---")
        st.write("**Sample transcripts**")
        if samples:
            selected = st.selectbox(
                "Load a sample", ["Select a sample..."] + list(samples.keys())
            )
            if selected != "Select a sample...":
                st.session_state.transcript_text = samples[selected]
                st.session_state.last_source = selected
        else:
            st.caption("No sample files found in sample_transcripts/")
        st.markdown("---")
        st.write("**Current source**")
        st.caption(st.session_state.last_source)


def main() -> None:
    init_state()
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    samples = load_sample_files()
    render_sidebar(samples)
    render_header()

    top_left, top_right = st.columns([1.35, 0.65], gap="large")

    with top_left:
        st.markdown(
            "<div class='section-title'>Meeting input</div>", unsafe_allow_html=True
        )
        st.markdown(
            "<div class='subsection-title'>Upload transcript</div>",
            unsafe_allow_html=True,
        )
        uploaded = st.file_uploader(
            "Upload transcript (.txt)", type=["txt"], label_visibility="collapsed"
        )
        if uploaded is not None:
            st.session_state.transcript_text = uploaded.read().decode(
                "utf-8", errors="ignore"
            )
            st.session_state.last_source = uploaded.name

        st.markdown(
            "<div class='subsection-title'>Paste transcript</div>",
            unsafe_allow_html=True,
        )
        st.text_area(
            "Paste transcript",
            key="transcript_text",
            height=300,
            placeholder="Paste meeting transcript here...",
            label_visibility="collapsed",
        )

        action_cols = st.columns([0.24, 0.24, 0.022, 0.24, 0.258])
        with action_cols[1]:
            extract_clicked = st.button(
                "Extract Actions", use_container_width=True, type="primary"
            )
        with action_cols[3]:
            if st.button("Clear Output", use_container_width=True):
                st.session_state.parsed = None
                st.session_state.edited_df = None

        if extract_clicked:
            if not can_call_model():
                st.error("Daily AI request limit reached.")
            else:
                with st.spinner("Extracting meeting intelligence..."):
                    st.session_state.parsed = extract_meeting_data(
                        st.session_state.transcript_text
                    )
                    st.session_state.edited_df = to_dataframe(st.session_state.parsed)

    with top_right:
        st.markdown("<div class='metrics-stack'>", unsafe_allow_html=True)
        parsed = st.session_state.parsed
        df = (
            st.session_state.edited_df
            if isinstance(st.session_state.edited_df, pd.DataFrame)
            else to_dataframe(parsed)
        )
        total_actions = len(df)
        open_actions = (
            len(df[df["status"].isin(["Not Started", "In Progress", "Blocked"])])
            if not df.empty
            else 0
        )
        high_priority = len(df[df["priority"] == "High"]) if not df.empty else 0
        decisions = len(parsed.get("decisions", [])) if parsed else 0

        metric_cols = st.columns(2, gap="medium")
        with metric_cols[0]:
            render_metric_card("Actions", total_actions)
        with metric_cols[1]:
            render_metric_card("Open Items", open_actions)

        st.markdown("<div style='height: 0.9rem;'></div>", unsafe_allow_html=True)

        metric_cols = st.columns(2, gap="medium")
        with metric_cols[0]:
            render_metric_card("High Priority", high_priority)
        with metric_cols[1]:
            render_metric_card("Decisions", decisions)

        model_note = (
            current_model_name() if get_client() is not None else "Fallback mode"
        )
        st.markdown(
            f"<div class='good-strip metric-status'><b>Model:</b> {model_note}<br><b>Source:</b> {st.session_state.last_source}</div>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    parsed = st.session_state.parsed
    if parsed:
        intelligence_tab, actions_tab, owners_tab, export_tab = st.tabs(
            [
                "Meeting Intelligence",
                "Action Register",
                "Owner Workload",
                "Export",
            ]
        )

        with intelligence_tab:
            st.markdown(
                "<div class='content-title'>Meeting overview</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div class='entity-title'>{parsed.get('meeting_title', 'Untitled')}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div class='content-date'>Date: {parsed.get('meeting_date', '')}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<div class='content-subtitle'>Executive summary</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div class='good-strip'>{parsed.get('summary', '')}</div>",
                unsafe_allow_html=True,
            )
            st.markdown("<div class='content-divider'></div>", unsafe_allow_html=True)
            st.markdown(
                "<div class='content-subtitle'>Decisions</div>", unsafe_allow_html=True
            )
            for item in parsed.get("decisions", []):
                st.markdown(f"- {item}")
            st.markdown("<div class='content-divider'></div>", unsafe_allow_html=True)
            st.markdown(
                "<div class='content-subtitle'>Risks / blockers</div>",
                unsafe_allow_html=True,
            )
            for item in parsed.get("risks_blockers", []):
                st.markdown(f"- {item}")

        with actions_tab:
            df = (
                st.session_state.edited_df
                if isinstance(st.session_state.edited_df, pd.DataFrame)
                else to_dataframe(parsed)
            )
            readiness_stats = analyze_execution_readiness(df)
            render_execution_readiness(readiness_stats)

            filter1, filter2 = st.columns([0.5, 0.5])
            with filter1:
                status_filter = st.multiselect(
                    "Filter by status",
                    ["Not Started", "In Progress", "Blocked", "Done"],
                    default=["Not Started", "In Progress", "Blocked", "Done"],
                )
            with filter2:
                priority_filter = st.multiselect(
                    "Filter by priority",
                    ["High", "Medium", "Low"],
                    default=["High", "Medium", "Low"],
                )

            filtered_df = (
                df[
                    df["status"].isin(status_filter)
                    & df["priority"].isin(priority_filter)
                ].copy()
                if not df.empty
                else df.copy()
            )

            edited = st.data_editor(
                filtered_df,
                use_container_width=True,
                num_rows="dynamic",
                hide_index=True,
                column_config={
                    "task": st.column_config.TextColumn(width="large"),
                    "owner": st.column_config.TextColumn(width="medium"),
                    "deadline": st.column_config.TextColumn(width="medium"),
                    "priority": st.column_config.SelectboxColumn(
                        options=["High", "Medium", "Low"]
                    ),
                    "status": st.column_config.SelectboxColumn(
                        options=["Not Started", "In Progress", "Blocked", "Done"]
                    ),
                    "notes": st.column_config.TextColumn(width="large"),
                },
                key="action_editor",
            )

            st.session_state.edited_df = merge_filtered_edits(df, filtered_df, edited)

        with owners_tab:
            df = (
                st.session_state.edited_df
                if isinstance(st.session_state.edited_df, pd.DataFrame)
                else to_dataframe(parsed)
            )
            owner_df = owner_summary(df)
            st.markdown(
                "<div class='panel-label'>Owner-wise task load</div>",
                unsafe_allow_html=True,
            )
            st.dataframe(owner_df, use_container_width=True, hide_index=True)

        with export_tab:
            st.markdown(
                "<div class='subsection-title'>Export outputs</div>",
                unsafe_allow_html=True,
            )
            df = (
                st.session_state.edited_df
                if isinstance(st.session_state.edited_df, pd.DataFrame)
                else to_dataframe(parsed)
            )
            brief_pdf = build_action_brief_pdf(parsed, df)
            email_draft = build_follow_up_email(parsed, df).encode("utf-8")
            tracker_csv = build_task_tracker_csv(parsed, df)

            st.download_button(
                "Download Action Brief (PDF)",
                brief_pdf,
                file_name="action_brief.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
            st.download_button(
                "Download Follow-up Email Draft",
                email_draft,
                file_name="follow_up_email_draft.txt",
                mime="text/plain",
                use_container_width=True,
            )
            st.download_button(
                "Export Task Tracker CSV (Notion / Google Sheets)",
                tracker_csv,
                file_name="task_tracker_notion_google_sheets.csv",
                mime="text/csv",
                use_container_width=True,
            )
    else:
        st.info(
            "No extraction yet. Load a sample or paste a transcript, then click Extract actions."
        )


if __name__ == "__main__":
    main()
