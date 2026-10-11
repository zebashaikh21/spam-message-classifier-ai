// Set this to the deployed FastAPI backend URL, e.g. https://your-api.onrender.com
const API_BASE = (window.API_BASE_URL || "").replace(/\/$/, "");
const $ = (id) => document.getElementById(id);
let latestBatch = [];

function setPage(name) {
  document.querySelectorAll(".page").forEach(p => p.classList.remove("active"));
  document.querySelectorAll(".nav-item").forEach(b => b.classList.toggle("active", b.dataset.page === name));
  $("page-" + name).classList.add("active");
  window.scrollTo({top:0,behavior:"smooth"});
}
document.querySelectorAll(".nav-item").forEach(b => b.addEventListener("click",()=>setPage(b.dataset.page)));
document.querySelectorAll("[data-goto]").forEach(b=>b.addEventListener("click",()=>setPage(b.dataset.goto)));

function showError(message) {
  const box=$("globalError"); box.textContent=message; box.classList.remove("hidden");
  setTimeout(()=>box.classList.add("hidden"),7000);
}
async function api(path, options={}) {
  if(!API_BASE) throw new Error("Backend API URL is not configured. Set API_BASE_URL in frontend/config.js to your deployed FastAPI URL.");
  const response=await fetch(API_BASE+path,options);
  const data=await response.json().catch(()=>({detail:"The API returned an invalid response."}));
  if(!response.ok) throw new Error(data.detail || "API request failed.");
  return data;
}
function fmt(n){return Number(n||0).toLocaleString("en-US")}
function pct(n){return (Number(n||0)*100).toFixed(1)+"%"}
function renderResult(target,result){
  const spam=result.label==="Spam";
  $(target).innerHTML=`<div class="result ${spam?"spam":"ham"}"><div><div class="result-title">${spam?"⚠ Spam":"✓ Not Spam"}</div><div class="result-sub">${spam?"This message has patterns associated with spam.":"This message looks more like a normal message."}</div></div><div class="confidence"><strong>${pct(result.confidence)}</strong><span>CONFIDENCE</span></div></div>`;
}
async function classifyMessage(inputId, outputId, buttonId) {
  const message=$(inputId).value.trim();
  if(!message){showError("Please enter a message before classifying.");return;}
  const button=buttonId?$(buttonId):null;
  if(button){button.disabled=true;button.dataset.original=button.textContent;button.textContent="Analysing…";}
  try{
    const result=await api("/predict",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({message})});
    renderResult(outputId,result);
  }catch(e){showError(e.message);}
  finally{if(button){button.disabled=false;button.textContent=button.dataset.original;}}
}
$("homeAnalyze").addEventListener("click",()=>classifyMessage("homeMessage","homeResult","homeAnalyze"));
$("predictButton").addEventListener("click",()=>classifyMessage("predictMessage","predictResult","predictButton"));
[["homeMessage","homeCount"],["predictMessage","predictCount"]].forEach(([i,c])=>$(i).addEventListener("input",()=>$(c).textContent=$(i).value.length+" / 5000"));

function metric(id,value){$(id).textContent=value}
function drawDonut(m){
  if(!window.Plotly)return;
  Plotly.newPlot("homeDonut",[{type:"pie",labels:["Not Spam","Spam"],values:[m.not_spam_messages,m.spam_messages],hole:.67,marker:{colors:["#20b56b","#f05a6d"]},textinfo:"none",hovertemplate:"%{label}: %{value:,}<extra></extra>"}],{margin:{t:5,b:5,l:5,r:5},showlegend:false,paper_bgcolor:"rgba(0,0,0,0)",annotations:[{text:"Messages",showarrow:false,font:{size:13,color:"#63718b"}}]},{displayModeBar:false,responsive:true});
}
function renderReport(report){
  const body=$("reportTable").querySelector("tbody");body.innerHTML="";
  [["Not Spam",report["Not Spam"]],["Spam",report["Spam"]]].forEach(([label,r])=>{
    if(!r)return;const tr=document.createElement("tr");
    tr.innerHTML=`<td>${label}</td><td>${Number(r.precision).toFixed(3)}</td><td>${Number(r.recall).toFixed(3)}</td><td>${Number(r["f1-score"]).toFixed(3)}</td><td>${fmt(r.support)}</td>`;body.appendChild(tr);
  });
}
async function loadDashboard(){
  try{
    const m=await api("/metrics");
    $("apiStatus").textContent="Classifier online";$("apiStatus").previousElementSibling.classList.add("ok");
    metric("homeTotal",fmt(m.total_messages));metric("homeSpam",fmt(m.spam_messages));metric("homeHam",fmt(m.not_spam_messages));metric("homeAccuracy",pct(m.accuracy));
    metric("legendHam",fmt(m.not_spam_messages));metric("legendSpam",fmt(m.spam_messages));
    metric("perfAccuracy",pct(m.accuracy));metric("perfPrecision",pct(m.precision_spam));metric("perfRecall",pct(m.recall_spam));metric("perfF1",pct(m.f1_spam));
    metric("dataTotal",fmt(m.total_messages));metric("dataSpam",fmt(m.spam_messages));metric("dataHam",fmt(m.not_spam_messages));
    drawDonut(m);
    const cm=m.confusion_matrix;
    Plotly.newPlot("confusionChart",[{type:"heatmap",z:cm,x:["Predicted Not Spam","Predicted Spam"],y:["Actual Not Spam","Actual Spam"],colorscale:[[0,"#eff6ff"],[.5,"#60a5fa"],[1,"#1d4ed8"]],showscale:true,text:cm.map(r=>r.map(String)),texttemplate:"%{text}",hovertemplate:"%{y}<br>%{x}: %{z}<extra></extra>"}],{margin:{t:15,b:60,l:95,r:10},paper_bgcolor:"rgba(0,0,0,0)",font:{color:"#263756"},xaxis:{tickfont:{size:10}},yaxis:{tickfont:{size:10}}},{displayModeBar:false,responsive:true});
    const report=m.classification_report||{};renderReport(report);
    const counts=[{label:"Not Spam",count:m.not_spam_messages},{label:"Spam",count:m.spam_messages}];
    Plotly.newPlot("datasetBar",[{type:"bar",x:counts.map(x=>x.label),y:counts.map(x=>x.count),marker:{color:["#2f6ff3","#f05a6d"]},text:counts.map(x=>fmt(x.count)),textposition:"outside"}],{margin:{t:20,b:40,l:45,r:15},paper_bgcolor:"rgba(0,0,0,0)",plot_bgcolor:"rgba(0,0,0,0)",yaxis:{gridcolor:"#edf1f8"},font:{color:"#263756"}},{displayModeBar:false,responsive:true});
  }catch(e){$("apiStatus").textContent="API not configured";showError(e.message);}
}
$("batchButton").addEventListener("click",async()=>{
  const file=$("csvFile").files[0];$("batchError").classList.add("hidden");
  if(!file){$("batchError").textContent="Choose a CSV file first."; $("batchError").classList.remove("hidden");return;}
  const button=$("batchButton");button.disabled=true;button.textContent="Classifying…";
  try{
    const form=new FormData();form.append("file",file);
    const data=await api("/predict-csv",{method:"POST",body:form});latestBatch=data.results;
    $("batchSummary").innerHTML=`<article class="metric-card"><div class="metric-top">Messages processed</div><div class="metric-value">${fmt(data.total)}</div></article><article class="metric-card"><div class="metric-top">Spam detected</div><div class="metric-value">${fmt(data.spam_count)}</div></article><article class="metric-card"><div class="metric-top">Not Spam</div><div class="metric-value">${fmt(data.not_spam_count)}</div></article>`;
    $("batchSummary").classList.remove("hidden");$("batchTablePanel").classList.remove("hidden");
    const tbody=$("batchTable").querySelector("tbody");tbody.innerHTML="";
    data.results.forEach((r,i)=>{const tr=document.createElement("tr");const msg=document.createElement("td");msg.textContent=r.message;const pred=document.createElement("td");pred.innerHTML=`<span class="badge ${r.label==="Spam"?"spam":"ham"}">${r.label}</span>`;tr.innerHTML=`<td>${i+1}</td>`;tr.append(msg,pred);const conf=document.createElement("td");conf.textContent=pct(r.confidence);tr.append(conf);tbody.appendChild(tr);});
  }catch(e){$("batchError").textContent=e.message;$("batchError").classList.remove("hidden");}
  finally{button.disabled=false;button.textContent="Classify file →";}
});
$("downloadCsv").addEventListener("click",()=>{
  const header=["message","prediction","confidence","spam_probability"];
  const rows=latestBatch.map(r=>[r.message,r.label,r.confidence,r.spam_probability]);
  const csv=[header,...rows].map(row=>row.map(v=>`"${String(v??"").replace(/"/g,'""')}"`).join(",")).join("\r\n");
  const url=URL.createObjectURL(new Blob([csv],{type:"text/csv;charset=utf-8;"}));const a=document.createElement("a");a.href=url;a.download="spam_predictions.csv";a.click();URL.revokeObjectURL(url);
});
loadDashboard();
