import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest,RandomForestClassifier
from sklearn.model_selection import train_test_split
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Image,PageBreak
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import A4
from reportlab.lib.enums import TA_CENTER

start=time.time()

BASE_DIR=os.path.dirname(os.path.abspath(__file__))

files=[f for f in os.listdir(BASE_DIR) if f.endswith(".csv") or f.endswith(".xlsx")]
INPUT=os.path.join(BASE_DIR,files[0])

OUT=os.path.join(BASE_DIR,"Output")
CHART=os.path.join(OUT,"charts")
MODEL=os.path.join(OUT,"models")

os.makedirs(CHART,exist_ok=True)
os.makedirs(MODEL,exist_ok=True)

if INPUT.endswith(".csv"):
    df=pd.read_csv(INPUT)
else:
    df=pd.read_excel(INPUT)

df.columns=[c.lower().replace(" ","_") for c in df.columns]

df=df.drop_duplicates()

tcol=None
for c in df.columns:
    if "time" in c or "date" in c:
        tcol=c
        break

if tcol is None:
    df["date"]=pd.date_range("2020-01-01",periods=len(df),freq="h")
    tcol="date"

df[tcol]=pd.to_datetime(df[tcol],errors="coerce")
df=df.dropna(subset=[tcol])
df=df.sort_values(tcol)
df=df.set_index(tcol)

df=df.drop(columns=[c for c in df.columns if "unnamed" in c],errors="ignore")

num=df.select_dtypes(include=np.number).columns.tolist()

df[num]=df[num].replace([np.inf,-np.inf],np.nan)
df[num]=df[num].fillna(df[num].median())

daily=df.resample("D").mean(numeric_only=True)
daily["incident_count"]=df.resample("D").size()
daily=daily.fillna(0)

weekly=daily.resample("W").mean()

units={
"sales":"USD",
"item_visibility":"visibility",
"item_weight":"weight",
"rating":"rating",
"outlet_establishment_year":"year",
"incident_count":"records"
}

def insight(s,n,u):
    m=round(s.mean(),2)
    mx=round(s.max(),2)
    mn=round(s.min(),2)
    sd=round(s.std(),2)
    return f"{n} averaged {m} {u}. Maximum reached {mx} {u} while minimum was {mn} {u}. Standard deviation {sd} reflects operational variability."

def save(fig,name):
    p=os.path.join(CHART,name+".png")
    fig.savefig(p,dpi=300,bbox_inches="tight")
    plt.close()
    return p

def tsplot(s,title,y,u,name):
    fig,ax=plt.subplots(figsize=(16,9))
    ax.plot(s.index,s.values,linewidth=2)
    ax.set_title(title,fontsize=22)
    ax.set_xlabel("Time")
    ax.set_ylabel(f"{y} ({u})")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
    ax.yaxis.set_major_locator(MaxNLocator(6))
    ax.grid(alpha=.3)
    fig.tight_layout(rect=[0,0,1,0.95])
    return save(fig,name)

charts=[]
texts=[]

charts.append(tsplot(weekly["incident_count"],"Record Frequency Over Time","Records","records","record_frequency"))
texts.append(insight(weekly["incident_count"],"Record frequency","records"))

core="sales" if "sales" in weekly.columns else num[0]

charts.append(tsplot(weekly[core],"Sales Trend",core,units.get(core,"value"),"sales_trend"))
texts.append(insight(weekly[core],core,units.get(core,"value")))

fig,ax=plt.subplots(figsize=(14,8))
sns.histplot(df[core],bins=40)
ax.set_title("Sales Distribution")
charts.append(save(fig,"sales_hist"))
texts.append("Distribution of sales values showing spread and concentration of revenue performance.")

corr=weekly.corr()
mask=np.triu(np.ones_like(corr,dtype=bool))

fig,ax=plt.subplots(figsize=(16,12))
sns.heatmap(corr,mask=mask,cmap="viridis",annot=False)
plt.xticks(rotation=45,ha="right",fontsize=9)
plt.yticks(fontsize=9)
ax.set_title("Operational Correlation Matrix")
charts.append(save(fig,"correlation_matrix"))
texts.append("Correlation matrix showing statistical relationships between operational indicators.")

feat=df[num].copy()
feat=feat.replace([np.inf,-np.inf],np.nan)
feat=feat.fillna(feat.median())
feat=feat.fillna(0)

kmeans=KMeans(n_clusters=4,n_init=10)
df["cluster"]=kmeans.fit_predict(feat)

fig,ax=plt.subplots(figsize=(12,7))
df["cluster"].value_counts().plot(kind="bar",ax=ax)
ax.set_title("Cluster Distribution")
charts.append(save(fig,"cluster_distribution"))
texts.append("Clustering groups records with similar operational characteristics.")

cluster_center=pd.DataFrame(kmeans.cluster_centers_,columns=feat.columns)

fig,ax=plt.subplots(figsize=(14,8))
sns.heatmap(cluster_center,cmap="coolwarm")
ax.set_title("Cluster Feature Heatmap")
charts.append(save(fig,"cluster_heatmap"))
texts.append("Heatmap illustrating feature influence across clusters.")

prov="outlet_type" if "outlet_type" in df.columns else None
serv="item_type" if "item_type" in df.columns else None

if prov:
    pr=df.groupby(prov)[core].mean()
    fig,ax=plt.subplots(figsize=(12,7))
    pr.plot(kind="bar",ax=ax)
    ax.set_title("Outlet Type Performance")
    charts.append(save(fig,"provider_reliability"))
    texts.append("Comparison of outlet types based on average sales.")

if serv:
    sr=df.groupby(serv)[core].mean().sort_values()
    top=sr.tail(20)
    fig,ax=plt.subplots(figsize=(12,8))
    top.plot(kind="barh",ax=ax)
    ax.set_title("Top 20 Item Type Sales Ranking")
    charts.append(save(fig,"service_ranking"))
    texts.append("Top twenty item categories ranked by average sales.")

df["risk_score"]=df[num].sum(axis=1)

fig,ax=plt.subplots(figsize=(14,8))
sns.histplot(df["risk_score"],bins=40)
ax.set_title("Risk Score Distribution")
charts.append(save(fig,"risk_distribution"))
texts.append("Composite operational score distribution.")

risk_week=df["risk_score"].resample("W").mean()
charts.append(tsplot(risk_week,"Risk Score Trend","Risk","score","risk_trend"))
texts.append(insight(risk_week,"Operational risk","score"))

df["severity"]=pd.qcut(df[core],3,labels=[0,1,2])
X=df[num]
y=df["severity"]

rf=RandomForestClassifier()
rf.fit(X,y)

imp=pd.Series(rf.feature_importances_,index=X.columns).sort_values()

fig,ax=plt.subplots(figsize=(12,7))
imp.plot(kind="barh",ax=ax)
ax.set_title("Sales Feature Importance")
charts.append(save(fig,"feature_importance"))
texts.append("Feature importance derived from sales prediction model.")

mttr=df[core].mean()
mtbf=(df.index.max()-df.index.min()).total_seconds()/3600/max(len(df),1)

scaler=StandardScaler()
scaled=scaler.fit_transform(weekly)
iso=IsolationForest(contamination=.02)
weekly["anomaly"]=iso.fit_predict(scaled)

ts=weekly["incident_count"]

model=ExponentialSmoothing(ts,trend="add")
fit=model.fit()
forecast=fit.forecast(12)

fig,ax=plt.subplots(figsize=(16,9))
ax.plot(ts,label="Historical")
ax.plot(forecast,label="Forecast")
ax.set_title("Record Forecast")
ax.legend()
charts.append(save(fig,"forecast"))
texts.append("Time series forecast predicting future record frequency.")

def executive(df):
    inc=len(df)
    avg=df[core].mean()
    mx=df[core].max()
    return f"""
Dataset contained {inc} records during the observation window.
Average sales value was {round(avg,2)} with maximum reaching {round(mx,2)}.
Operational patterns indicate measurable variability in sales performance.
Clustering highlights groups of products with similar characteristics.
Outlet and product category comparisons reveal structural differences in revenue generation.
"""

styles=getSampleStyleSheet()

title_style=ParagraphStyle(
"name='title'",
fontSize=30,
leading=36,
alignment=TA_CENTER,
textColor=HexColor("#22d3ee"),
spaceAfter=20
)

subtitle_style=ParagraphStyle(
"name='subtitle'",
fontSize=18,
leading=24,
alignment=TA_CENTER,
textColor=HexColor("#a78bfa"),
spaceAfter=40
)

body_style=ParagraphStyle(
"name='body'",
fontSize=14,
leading=20,
spaceAfter=20
)

doc=SimpleDocTemplate(
os.path.join(OUT,"Retail_Sales_Report.pdf"),
pagesize=A4,
leftMargin=72,
rightMargin=72,
topMargin=72,
bottomMargin=72
)

elements=[]

elements.append(Paragraph("BlinkIT Retail Analytics",title_style))
elements.append(Paragraph("Sales Intelligence Report",subtitle_style))

summary=f"""
Total Records Analysed: {len(df)}<br/>
Observation Window: {df.index.min()} to {df.index.max()}<br/>
Average Sales: {round(mttr,2)}<br/>
Data Frequency Interval: {round(mtbf,2)} hours
"""

elements.append(Paragraph(summary,body_style))
elements.append(Paragraph(executive(df),body_style))
elements.append(PageBreak())

for c,t in zip(charts,texts):
    elements.append(Image(c,width=6.3*inch,height=4.2*inch))
    elements.append(Spacer(1,35))
    elements.append(Paragraph(t,body_style))
    elements.append(PageBreak())

doc.build(elements)

print("Execution Time:",round(time.time()-start,2),"seconds")