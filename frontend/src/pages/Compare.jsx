import React, { useState, useRef, useEffect, useCallback } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Upload, Plus, Trash2, Download, Scale,
  ChevronLeft, ChevronRight, Loader2,
  CheckCircle, AlertCircle, XCircle,
} from "lucide-react";

// =====================================================
// V4 COMPARE ENGINE
// =====================================================

const TEMPLATE_W = 512;
const TEMPLATE_H = 716;
const REGIONS = {
  nameHP: {x:0.09,y:0.07,w:0.80,h:0.11},
  art:    {x:0.14,y:0.20,w:0.72,h:0.33},
  text:   {x:0.12,y:0.58,w:0.76,h:0.20},
  bottom: {x:0.12,y:0.86,w:0.76,h:0.05},
  border: {x:0.03,y:0.03,w:0.94,h:0.94}
};

function clamp(v,a,b){ return Math.max(a, Math.min(b,v)); }

function normalizeCard(img, doBrightness){
  const tmp = document.createElement('canvas'); tmp.width = img.width; tmp.height = img.height;
  const tctx = tmp.getContext('2d'); tctx.drawImage(img,0,0);
  const src = tctx.getImageData(0,0,tmp.width,tmp.height);
  const box = detectCardBounds(src);
  const out = document.createElement('canvas'); out.width = TEMPLATE_W; out.height = TEMPLATE_H;
  const octx = out.getContext('2d');
  octx.fillStyle = '#f7e3bd'; octx.fillRect(0,0,TEMPLATE_W,TEMPLATE_H);
  octx.drawImage(tmp, box.x, box.y, box.w, box.h, 0, 0, TEMPLATE_W, TEMPLATE_H);
  let imageData = octx.getImageData(0,0,TEMPLATE_W,TEMPLATE_H);
  if(doBrightness) imageData = brightnessNormalize(imageData);
  return { imageData };
}

function detectCardBounds(img){
  const {width:w,height:h,data} = img;
  let minX=w, minY=h, maxX=0, maxY=0;
  const samples = [0,4*(w-1),4*((h-1)*w),4*((h*w)-1)];
  let br=0,bg=0,bb=0;
  for(const i of samples){ br+=data[i]; bg+=data[i+1]; bb+=data[i+2]; }
  br/=4; bg/=4; bb/=4;
  for(let y=0;y<h;y++) for(let x=0;x<w;x++){
    const i=(y*w+x)*4;
    if(Math.abs(data[i]-br)+Math.abs(data[i+1]-bg)+Math.abs(data[i+2]-bb)>70){
      if(x<minX)minX=x; if(y<minY)minY=y; if(x>maxX)maxX=x; if(y>maxY)maxY=y;
    }
  }
  if(minX>=maxX||minY>=maxY) return {x:0,y:0,w,h};
  const padX=Math.round((maxX-minX)*0.03), padY=Math.round((maxY-minY)*0.03);
  const x1=clamp(minX-padX,0,w-1), y1=clamp(minY-padY,0,h-1);
  const x2=clamp(maxX+padX,0,w-1), y2=clamp(maxY+padY,0,h-1);
  return {x:x1,y:y1,w:x2-x1+1,h:y2-y1+1};
}

function brightnessNormalize(img){
  const out = new ImageData(new Uint8ClampedArray(img.data), img.width, img.height);
  let mean=0; const n=img.width*img.height;
  for(let i=0;i<out.data.length;i+=4) mean+=(0.299*out.data[i]+0.587*out.data[i+1]+0.114*out.data[i+2]);
  mean/=n; const gain=145/Math.max(1,mean);
  for(let i=0;i<out.data.length;i+=4){
    out.data[i]=clamp(out.data[i]*gain,0,255);
    out.data[i+1]=clamp(out.data[i+1]*gain,0,255);
    out.data[i+2]=clamp(out.data[i+2]*gain,0,255);
  }
  return out;
}

function cropRegion(img, r){
  const x=Math.round(r.x*img.width), y=Math.round(r.y*img.height);
  const w=Math.round(r.w*img.width), h=Math.round(r.h*img.height);
  const out = new ImageData(w,h);
  for(let yy=0;yy<h;yy++) for(let xx=0;xx<w;xx++){
    const si=((y+yy)*img.width+(x+xx))*4, di=(yy*w+xx)*4;
    out.data[di]=img.data[si]; out.data[di+1]=img.data[si+1];
    out.data[di+2]=img.data[si+2]; out.data[di+3]=255;
  }
  return out;
}

function grayscaleArray(img){
  const out=new Float32Array(img.width*img.height);
  for(let i=0,j=0;i<img.data.length;i+=4,j++) out[j]=0.299*img.data[i]+0.587*img.data[i+1]+0.114*img.data[i+2];
  return out;
}

function quickCorrelation(a,b){
  const ga=grayscaleArray(a), gb=grayscaleArray(b);
  let ma=0,mb=0; for(let i=0;i<ga.length;i++){ma+=ga[i];mb+=gb[i]} ma/=ga.length; mb/=gb.length;
  let num=0,da=0,db=0;
  for(let i=0;i<ga.length;i++){const xa=ga[i]-ma,xb=gb[i]-mb;num+=xa*xb;da+=xa*xa;db+=xb*xb;}
  return num/Math.sqrt(Math.max(1e-6,da*db));
}

function spatialScore(a,b){
  const ga=grayscaleArray(a), gb=grayscaleArray(b);
  let diff=0; for(let i=0;i<ga.length;i++) diff+=Math.abs(ga[i]-gb[i]);
  return clamp(100-(diff/(ga.length*255))*100,0,100);
}

function sharpnessValue(img){
  const g=grayscaleArray(img), w=img.width, h=img.height;
  let sum=0,n=0;
  for(let y=1;y<h-1;y++) for(let x=1;x<w-1;x++){
    const i=y*w+x; sum+=Math.abs(4*g[i]-g[i-1]-g[i+1]-g[i-w]-g[i+w]); n++;
  }
  return sum/Math.max(1,n);
}

function sharpnessMatch(ref,cmp){
  const a=sharpnessValue(ref), b=sharpnessValue(cmp);
  return clamp(Math.min(a,b)/Math.max(1,Math.max(a,b))*100,0,100);
}
function borderConsistency(ref,cmp){
  const strips=[{x:0.03,y:0.03,w:0.94,h:0.04},{x:0.03,y:0.93,w:0.94,h:0.04},{x:0.03,y:0.03,w:0.04,h:0.94},{x:0.93,y:0.03,w:0.04,h:0.94}];
  let scores=[]; for(const s of strips) scores.push(spatialScore(cropRegion(ref,s),cropRegion(cmp,s)));
  return scores.reduce((a,b)=>a+b,0)/scores.length;
}

function highPassSignature(img){
  const g=grayscaleArray(img), w=img.width, h=img.height;
  const bins=new Float32Array(8), counts=new Float32Array(8);
  for(let y=1;y<h-1;y++) for(let x=1;x<w-1;x++){
    const i=y*w+x;
    const gx=-g[i-w-1]-2*g[i-1]-g[i+w-1]+g[i-w+1]+2*g[i+1]+g[i+w+1];
    const gy=-g[i-w-1]-2*g[i-w]-g[i-w+1]+g[i+w-1]+2*g[i+w]+g[i+w+1];
    const mag=Math.sqrt(gx*gx+gy*gy)/255;
    const bin=Math.min(7,Math.floor(Math.sqrt(((x/(w-1))-0.5)**2+((y/(h-1))-0.5)**2)*8));
    bins[bin]+=mag; counts[bin]++;
  }
  for(let i=0;i<bins.length;i++) bins[i]=counts[i]?bins[i]/counts[i]:0;
  const wt=[0.15,0.4,0.8,1.0,1.15,1.2,1.2,1.1];
  for(let i=0;i<bins.length;i++) bins[i]*=wt[i];
  let sum=0; for(let i=0;i<bins.length;i++) sum+=bins[i];
  for(let i=0;i<bins.length;i++) bins[i]/=Math.max(sum,1e-6);
  return bins;
}

function fftRegionScore(a,b){
  const s1=highPassSignature(a), s2=highPassSignature(b);
  let diff=0; for(let i=0;i<s1.length;i++) diff+=Math.abs(s1[i]-s2[i]);
  return clamp(100-diff*140,0,100);
}

function microRegionScore(nameRef,nameCmp,artRef,artCmp,textRef,textCmp,bottomRef,bottomCmp){
  return 0.22*(sharpnessMatch(nameRef,nameCmp)*0.55+spatialScore(nameRef,nameCmp)*0.45)
    +0.28*(sharpnessMatch(artRef,artCmp)*0.35+spatialScore(artRef,artCmp)*0.65)
    +0.32*(sharpnessMatch(textRef,textCmp)*0.6+spatialScore(textRef,textCmp)*0.4)
    +0.18*(sharpnessMatch(bottomRef,bottomCmp)*0.7+spatialScore(bottomRef,bottomCmp)*0.3);
}

function dotPatternRegion(a,b){
  const da=highPassSignature(a), db=highPassSignature(b);
  let diff=0; for(let i=0;i<da.length;i++) diff+=Math.abs(da[i]-db[i]);
  return clamp(100-diff*120,0,100);
}

function dotPatternScore(...regions){
  const pairs=[]; for(let i=0;i<regions.length;i+=2) pairs.push([regions[i],regions[i+1]]);
  let vals=[]; for(const [a,b] of pairs) vals.push(dotPatternRegion(a,b));
  return vals.reduce((x,y)=>x+y,0)/vals.length;
}

function shiftImage(img, dx, dy){
  const out=new ImageData(img.width,img.height);
  const bg=[247,227,189,255];
  for(let i=0;i<out.data.length;i+=4) out.data.set(bg,i);
  for(let y=0;y<img.height;y++) for(let x=0;x<img.width;x++){
    const sx=x-dx, sy=y-dy;
    if(sx<0||sy<0||sx>=img.width||sy>=img.height) continue;
    const si=(sy*img.width+sx)*4, di=(y*img.width+x)*4;
    out.data[di]=img.data[si]; out.data[di+1]=img.data[si+1];
    out.data[di+2]=img.data[si+2]; out.data[di+3]=255;
  }
  return out;
}

function alignCompared(ref, cmp, radius){
  let best={score:-1,shiftX:0,shiftY:0,imageData:cmp};
  const regionRef=cropRegion(ref,REGIONS.art);
  for(let dy=-radius;dy<=radius;dy++) for(let dx=-radius;dx<=radius;dx++){
    const shifted=shiftImage(cmp,dx,dy);
    const s=quickCorrelation(regionRef,cropRegion(shifted,REGIONS.art));
    if(s>best.score) best={score:s,shiftX:dx,shiftY:dy,imageData:shifted};
  }
  return best;
}

function buildHeatmap(ref, cmp){
  const out=new ImageData(ref.width,ref.height);
  for(let i=0;i<ref.data.length;i+=4){
    const d=(Math.abs(ref.data[i]-cmp.data[i])+Math.abs(ref.data[i+1]-cmp.data[i+1])+Math.abs(ref.data[i+2]-cmp.data[i+2]))/3;
    out.data[i]=clamp(d*2.4,0,255); out.data[i+1]=20;
    out.data[i+2]=clamp(255-d*1.3,0,255); out.data[i+3]=255;
  }
  return out;
}

function runV4Comparison(refImg, cmpImg, alignRadius, doBrightness){
  const ref=normalizeCard(refImg,doBrightness);
  const cmp=normalizeCard(cmpImg,doBrightness);
  const aligned=alignCompared(ref.imageData,cmp.imageData,alignRadius);
  const heatmap=buildHeatmap(ref.imageData,aligned.imageData);
  const artRef=cropRegion(ref.imageData,REGIONS.art), artCmp=cropRegion(aligned.imageData,REGIONS.art);
  const nameRef=cropRegion(ref.imageData,REGIONS.nameHP), nameCmp=cropRegion(aligned.imageData,REGIONS.nameHP);
  const textRef=cropRegion(ref.imageData,REGIONS.text), textCmp=cropRegion(aligned.imageData,REGIONS.text);
  const bottomRef=cropRegion(ref.imageData,REGIONS.bottom), bottomCmp=cropRegion(aligned.imageData,REGIONS.bottom);
  const artFft=fftRegionScore(artRef,artCmp);
  const artSpatial=spatialScore(artRef,artCmp);
  const border=borderConsistency(ref.imageData,aligned.imageData);
  const textScore=spatialScore(textRef,textCmp)*0.55+sharpnessMatch(textRef,textCmp)*0.45;
  const micro=microRegionScore(nameRef,nameCmp,artRef,artCmp,textRef,textCmp,bottomRef,bottomCmp);
  const dot=dotPatternScore(artRef,artCmp,nameRef,nameCmp,textRef,textCmp);
  const sharpDelta=sharpnessMatch(nameRef,nameCmp)*0.45+sharpnessMatch(textRef,textCmp)*0.55;
  const finalScore=clamp(0.16*artFft+0.12*artSpatial+0.18*border+0.18*textScore+0.18*micro+0.14*dot+0.04*spatialScore(ref.imageData,aligned.imageData),0,100);
  let verdict='Suspicious', verdictType='warn';
  if(finalScore>=91&&dot>=88&&textScore>=88&&sharpDelta>=86){verdict='Likely real';verdictType='good';}
  else if(finalScore<82||dot<75||textScore<78||sharpDelta<76){verdict='Likely fake';verdictType='bad';}
  return {
    finalScore, verdict, verdictType, artFft, artSpatial, border, textScore, micro, dot, sharpDelta,
    bestShift:`(${aligned.shiftX}, ${aligned.shiftY})`,
    heatmap, refNorm:ref.imageData, cmpNorm:aligned.imageData,
    microRegions:{nameRef,nameCmp,artRef,artCmp,textRef,textCmp,bottomRef,bottomCmp}
  };
}
// =====================================================
// IMAGE VIEWER COMPONENT
// =====================================================

const SCALE_MIN = 0.2;
const SCALE_MAX = 8;
function sliderToScale(v){ return SCALE_MIN*Math.pow(SCALE_MAX/SCALE_MIN,parseFloat(v)/100); }
function scaleToSlider(s){ return Math.round((100*Math.log(Math.max(s,SCALE_MIN)/SCALE_MIN))/Math.log(SCALE_MAX/SCALE_MIN)); }

function ImageViewer({ label, image, state, onStateChange, canvasRef, onFileSelect }) {
  const wrapRef = useRef(null);
  const dragRef = useRef({ dragging: false, lastX: 0, lastY: 0 });
  const touchRef = useRef({ lastTouches: [], lastDist: 0 });
  const fileRef = useRef(null);

  const render = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!image) return;
    ctx.save();
    ctx.translate(state.ox, state.oy);
    ctx.scale(state.scale, state.scale);
    ctx.drawImage(image, 0, 0);
    ctx.restore();
  }, [image, state, canvasRef]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const wrap = wrapRef.current;
    if (!canvas || !wrap) return;
    canvas.width = wrap.clientWidth;
    canvas.height = wrap.clientHeight;
    render();
  }, [render, canvasRef]);

  useEffect(() => { render(); }, [render]);

  useEffect(() => {
    const handleResize = () => {
      const canvas = canvasRef.current;
      const wrap = wrapRef.current;
      if (!canvas || !wrap) return;
      canvas.width = wrap.clientWidth;
      canvas.height = wrap.clientHeight;
      render();
    };
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, [render, canvasRef]);

  const handleMouseDown = (e) => {
    dragRef.current = { dragging: true, lastX: e.clientX, lastY: e.clientY };
    canvasRef.current.style.cursor = "grabbing";
  };

  useEffect(() => {
    const handleMouseMove = (e) => {
      if (!dragRef.current.dragging) return;
      const dx = e.clientX - dragRef.current.lastX;
      const dy = e.clientY - dragRef.current.lastY;
      dragRef.current.lastX = e.clientX;
      dragRef.current.lastY = e.clientY;
      onStateChange((prev) => ({ ...prev, ox: prev.ox + dx, oy: prev.oy + dy }));
    };
    const handleMouseUp = () => {
      dragRef.current.dragging = false;
      if (canvasRef.current) canvasRef.current.style.cursor = "grab";
    };
    window.addEventListener("mousemove", handleMouseMove);
    window.addEventListener("mouseup", handleMouseUp);
    return () => {
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mouseup", handleMouseUp);
    };
  }, [onStateChange, canvasRef]);

  const handleWheel = (e) => {
    e.preventDefault();
    const rect = canvasRef.current.getBoundingClientRect();
    const cx = e.clientX - rect.left, cy = e.clientY - rect.top;
    const delta = e.deltaY > 0 ? 0.95 : 1.05;
    onStateChange((prev) => {
      const newScale = Math.min(SCALE_MAX, Math.max(SCALE_MIN, prev.scale * delta));
      const ratio = newScale / prev.scale;
      return { scale: newScale, ox: cx - ratio * (cx - prev.ox), oy: cy - ratio * (cy - prev.oy) };
    });
  };

  const getTouchDist = (t) => {
    const dx=t[0].clientX-t[1].clientX, dy=t[0].clientY-t[1].clientY;
    return Math.sqrt(dx*dx+dy*dy);
  };

  const handleTouchStart = (e) => {
    e.preventDefault();
    touchRef.current.lastTouches = Array.from(e.touches);
    if(e.touches.length===2) touchRef.current.lastDist=getTouchDist(e.touches);
  };

  const handleTouchMove = (e) => {
    e.preventDefault();
    const touches = Array.from(e.touches);
    const last = touchRef.current.lastTouches;
    if(touches.length===1 && last.length>=1){
      const dx=touches[0].clientX-last[0].clientX, dy=touches[0].clientY-last[0].clientY;
      onStateChange((prev)=>({...prev,ox:prev.ox+dx,oy:prev.oy+dy}));
    } else if(touches.length===2){
      const dist=getTouchDist(touches);
      if(touchRef.current.lastDist>0){
        const ratio=dist/touchRef.current.lastDist;
        const midX=(touches[0].clientX+touches[1].clientX)/2;
        const midY=(touches[0].clientY+touches[1].clientY)/2;
        const rect=canvasRef.current.getBoundingClientRect();
        const cx=midX-rect.left, cy=midY-rect.top;
        const cr=Math.min(1.06,Math.max(0.94,ratio));
        onStateChange((prev)=>{
          const ns=Math.min(SCALE_MAX,Math.max(SCALE_MIN,prev.scale*cr));
          const sr=ns/prev.scale;
          return{scale:ns,ox:cx-sr*(cx-prev.ox),oy:cy-sr*(cy-prev.oy)};
        });
      }
      touchRef.current.lastDist=dist;
    }
    touchRef.current.lastTouches=touches;
  };

  const handleTouchEnd = (e) => {
    touchRef.current.lastTouches=Array.from(e.touches);
    if(e.touches.length<2) touchRef.current.lastDist=0;
  };

  const handleSlider = (e) => {
    const newScale=sliderToScale(e.target.value);
    const canvas=canvasRef.current;
    const cx=canvas.width/2, cy=canvas.height/2;
    onStateChange((prev)=>{
      const ratio=newScale/prev.scale;
      return{scale:newScale,ox:cx-ratio*(cx-prev.ox),oy:cy-ratio*(cy-prev.oy)};
    });
  };

  const handleFile = (e) => {
    const file=e.target.files[0]; if(!file) return;
    const reader=new FileReader();
    reader.onload=(ev)=>{
      const img=new Image();
      img.onload=()=>{
        const canvas=canvasRef.current; if(!canvas) return;
        const fs=Math.min(canvas.width/img.width,canvas.height/img.height);
        onStateChange({scale:fs,ox:(canvas.width-img.width*fs)/2,oy:(canvas.height-img.height*fs)/2});
        onFileSelect(img);
      };
      img.src=ev.target.result;
    };
    reader.readAsDataURL(file); e.target.value="";
  };

  return (
    <div className="space-y-1">
      <div ref={wrapRef} className="relative w-full bg-card border border-border rounded-lg overflow-hidden"
        style={{ height: 280, touchAction: "none" }}>
        <canvas ref={canvasRef} className="absolute top-0 left-0 w-full h-full"
          style={{ cursor: image ? "grab" : "default" }}
          onMouseDown={handleMouseDown} onWheel={handleWheel}
          onTouchStart={handleTouchStart} onTouchMove={handleTouchMove} onTouchEnd={handleTouchEnd} />
        {!image && (
          <div className="absolute inset-0 flex items-center justify-center">
            <button onClick={() => fileRef.current?.click()}
              className="flex flex-col items-center gap-2 text-muted-foreground hover:text-foreground transition-colors cursor-pointer bg-transparent border-none shadow-none p-0 focus:outline-none focus:ring-0 active:bg-transparent">
              <Upload className="w-8 h-8" /><span className="text-sm">Upload image</span>
            </button>
          </div>
        )}
        {image && (
          <div className="absolute right-2 top-2 bottom-2 flex items-center">
            <input type="range" min="0" max="100" step="1" value={scaleToSlider(state.scale)}
              onChange={handleSlider} className="h-full opacity-30 hover:opacity-70 transition-opacity"
              style={{writingMode:"vertical-lr",direction:"rtl",width:18,WebkitAppearance:"slider-vertical",cursor:"pointer"}} />
          </div>
        )}
      </div>
      <div className="flex items-center justify-between px-1">
        <span className="text-sm font-semibold text-base-color">{label}</span>
        {image && <Button variant="ghost" size="sm" onClick={() => fileRef.current?.click()}
          className="h-7 px-2 text-xs text-muted-foreground">Replace</Button>}
      </div>
      <input ref={fileRef} type="file" accept="image/*" className="hidden" onChange={handleFile} />
    </div>
  );
}
// =====================================================
// HELPERS
// =====================================================

function CanvasDisplay({ imageData, label }) {
  const ref = useRef(null);
  useEffect(() => {
    if (!ref.current || !imageData) return;
    const c = ref.current;
    c.width = imageData.width; c.height = imageData.height;
    c.getContext("2d").putImageData(imageData, 0, 0);
  }, [imageData]);
  return (
    <div>
      {label && <p className="text-xs text-muted-foreground mb-1">{label}</p>}
      <canvas ref={ref} className="w-full rounded border border-border" />
    </div>
  );
}

function StatBox({ label, value, subtitle }) {
  return (
    <div className="bg-card border border-border rounded-lg p-3">
      <p className="text-xs text-muted-foreground uppercase tracking-wide font-semibold">{label}</p>
      <p className="text-xl font-bold mt-1">{value}</p>
      {subtitle && <p className="text-xs text-muted-foreground mt-1">{subtitle}</p>}
    </div>
  );
}

// =====================================================
// COMPARE TAB
// =====================================================

function createBlankEntry() {
  return { imgQ: null, imgR: null, stateQ: { scale:1,ox:0,oy:0 }, stateR: { scale:1,ox:0,oy:0 }, notes: "", filename: "", results: null };
}

function CompareTab() {
  const [entries, setEntries] = useState([createBlankEntry()]);
  const [currentIdx, setCurrentIdx] = useState(0);
  const [computing, setComputing] = useState(false);
  const [alignRadius, setAlignRadius] = useState(12);
  const [doBrightness, setDoBrightness] = useState(true);
  const [showDeleteModal, setShowDeleteModal] = useState(false);

  const canvasQRef = useRef(null);
  const canvasRRef = useRef(null);
  const entry = entries[currentIdx];

  const updateEntry = (field, value) => {
    setEntries((prev) => { const next=[...prev]; next[currentIdx]={...next[currentIdx],[field]:value}; return next; });
  };

  const handleCompare = () => {
    if (!entry.imgQ || !entry.imgR) return;
    setComputing(true);
    setTimeout(() => {
      try { updateEntry("results", runV4Comparison(entry.imgQ, entry.imgR, alignRadius, doBrightness)); }
      catch (err) { console.error("Comparison failed:", err); }
      setComputing(false);
    }, 50);
  };

  const addEntry = () => { setEntries((prev) => [...prev, createBlankEntry()]); setCurrentIdx(entries.length); };
  const deleteEntry = () => {
    setShowDeleteModal(false);
    if(entries.length===1){setEntries([createBlankEntry()]);setCurrentIdx(0);}
    else{setEntries((prev)=>{const next=[...prev];next.splice(currentIdx,1);return next;});setCurrentIdx((prev)=>Math.min(prev,entries.length-2));}
  };

  const r = entry.results;
  const verdictColor = r?.verdictType==='good'?'text-green-600':r?.verdictType==='bad'?'text-red-600':'text-amber-600';
  const VerdictIcon = r?.verdictType==='good'?CheckCircle:r?.verdictType==='bad'?XCircle:AlertCircle;

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <ImageViewer label="Reference document" image={entry.imgQ} state={entry.stateQ}
          onStateChange={(u)=>{setEntries((prev)=>{const next=[...prev];next[currentIdx]={...next[currentIdx],stateQ:typeof u==="function"?u(next[currentIdx].stateQ):u};return next;});}}
          canvasRef={canvasQRef} onFileSelect={(img)=>{updateEntry("imgQ",img);updateEntry("results",null);}} />
        <ImageViewer label="Compared document" image={entry.imgR} state={entry.stateR}
          onStateChange={(u)=>{setEntries((prev)=>{const next=[...prev];next[currentIdx]={...next[currentIdx],stateR:typeof u==="function"?u(next[currentIdx].stateR):u};return next;});}}
          canvasRef={canvasRRef} onFileSelect={(img)=>{updateEntry("imgR",img);updateEntry("results",null);}} />
      </div>

      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="flex items-center gap-2 text-base">
            <Scale className="w-5 h-5 text-purple-600" /> Document comparison
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-muted-foreground">Alignment radius</span>
                  <span className="font-mono font-medium">{alignRadius} px</span>
                </div>
                <input type="range" min="0" max="24" value={alignRadius}
                  onChange={(e)=>setAlignRadius(parseInt(e.target.value,10))} className="w-full accent-blue-600" />
              </div>
              <div>
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-muted-foreground">Brightness normalization</span>
                  <span className="font-mono font-medium">{doBrightness?"On":"Off"}</span>
                </div>
                <input type="range" min="0" max="1" step="1" value={doBrightness?1:0}
                  onChange={(e)=>setDoBrightness(e.target.value==="1")} className="w-full accent-blue-600" />
              </div>
            </div>
            <div className="flex items-center gap-4">
              <Button onClick={handleCompare} disabled={!entry.imgQ||!entry.imgR||computing} className="bg-blue-600 hover:bg-blue-700 text-white">
                {computing?<Loader2 className="w-4 h-4 mr-2 animate-spin"/>:<Scale className="w-4 h-4 mr-2"/>}
                {computing?"Analyzing...":"Compare documents"}
              </Button>
              {(!entry.imgQ||!entry.imgR)&&<p className="text-xs text-muted-foreground">Upload images to both panels.</p>}
            </div>
          </div>
        </CardContent>
      </Card>

      {r && (
        <>
          <Card>
            <CardContent className="pt-6">
              <div className="flex items-center gap-3 mb-4">
                <VerdictIcon className={`w-8 h-8 ${verdictColor}`} />
                <div>
                  <p className={`text-2xl font-bold ${verdictColor}`}>{r.verdict}</p>
                  <p className="text-sm text-muted-foreground">Final score: {r.finalScore.toFixed(1)}</p>
                </div>
              </div>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                <StatBox label="Artwork FFT" value={r.artFft.toFixed(1)} subtitle="Mid/high frequency" />
                <StatBox label="Artwork spatial" value={r.artSpatial.toFixed(1)} subtitle="Aligned similarity" />
                <StatBox label="Border" value={r.border.toFixed(1)} subtitle="Edge consistency" />
                <StatBox label="Text / stats" value={r.textScore.toFixed(1)} subtitle="Text area match" />
                <StatBox label="Micro analysis" value={r.micro.toFixed(1)} subtitle="Zoomed regions" />
                <StatBox label="Dot pattern" value={r.dot.toFixed(1)} subtitle="Print texture" />
                <StatBox label="Sharpness" value={r.sharpDelta.toFixed(1)} subtitle="Text crispness" />
                <StatBox label="Best shift" value={r.bestShift} subtitle="Alignment offset" />
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-3"><CardTitle className="text-base">Difference heatmap</CardTitle></CardHeader>
            <CardContent>
              <CanvasDisplay imageData={r.heatmap} />
              <p className="text-xs text-muted-foreground mt-2">Blue = similar. Red = different.</p>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-3"><CardTitle className="text-base">Normalized images</CardTitle></CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 gap-4">
                <CanvasDisplay imageData={r.refNorm} label="Reference (normalized)" />
                <CanvasDisplay imageData={r.cmpNorm} label="Compared (normalized)" />
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-3"><CardTitle className="text-base">Micro region analysis</CardTitle></CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <CanvasDisplay imageData={r.microRegions.nameRef} label="Name/HP (ref)" />
                <CanvasDisplay imageData={r.microRegions.nameCmp} label="Name/HP (cmp)" />
                <CanvasDisplay imageData={r.microRegions.artRef} label="Art (ref)" />
                <CanvasDisplay imageData={r.microRegions.artCmp} label="Art (cmp)" />
                <CanvasDisplay imageData={r.microRegions.textRef} label="Text (ref)" />
                <CanvasDisplay imageData={r.microRegions.textCmp} label="Text (cmp)" />
                <CanvasDisplay imageData={r.microRegions.bottomRef} label="Bottom (ref)" />
                <CanvasDisplay imageData={r.microRegions.bottomCmp} label="Bottom (cmp)" />
              </div>
            </CardContent>
          </Card>
        </>
      )}

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div>
          <label className="text-sm font-medium text-muted-foreground block mb-1">Notes (optional)</label>
          <textarea value={entry.notes} onChange={(e)=>updateEntry("notes",e.target.value)} placeholder="Add observations..."
            className="w-full min-h-[80px] border border-border rounded-lg p-3 text-sm bg-card text-base-color resize-vertical" />
        </div>
        <div>
          <label className="text-sm font-medium text-muted-foreground block mb-1">Case / file name</label>
          <input type="text" value={entry.filename} onChange={(e)=>updateEntry("filename",e.target.value)} placeholder="Enter case name..."
            className="w-full border border-border rounded-lg p-3 text-sm bg-card text-base-color" />
        </div>
        <div className="flex flex-col justify-end">
          <Button className="bg-blue-600 hover:bg-blue-700 text-white w-full">
            <Download className="w-4 h-4 mr-2" />Download report
          </Button>
        </div>
      </div>

      <div className="flex items-center justify-between py-2">
        <div className="flex gap-2">
          <Button variant="outline" size="sm" className="border-border text-foreground hover:bg-muted" onClick={addEntry}><Plus className="w-4 h-4 mr-1" /> New</Button>
          <Button variant="outline" size="sm" className="border-border text-foreground hover:bg-muted" onClick={()=>setShowDeleteModal(true)}><Trash2 className="w-4 h-4 mr-1" /> Delete</Button>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="outline" size="sm" onClick={()=>setCurrentIdx((i)=>Math.max(0,i-1))} disabled={currentIdx<=0}><ChevronLeft className="w-4 h-4" /></Button>
          <span className="text-sm text-muted-foreground">{currentIdx+1} / {entries.length}</span>
          <Button variant="outline" size="sm" onClick={()=>setCurrentIdx((i)=>Math.min(entries.length-1,i+1))} disabled={currentIdx>=entries.length-1}><ChevronRight className="w-4 h-4" /></Button>
        </div>
      </div>

      {showDeleteModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
          <div className="bg-card border border-border rounded-xl p-6 max-w-sm w-full mx-4 space-y-4">
            <h3 className="text-lg font-semibold">Delete entry</h3>
            <p className="text-sm text-muted-foreground">Are you sure? This cannot be undone.</p>
            <div className="flex gap-3 justify-end">
              <Button variant="outline" onClick={()=>setShowDeleteModal(false)}>Cancel</Button>
              <Button onClick={deleteEntry} className="bg-destructive hover:bg-destructive/90 text-destructive-foreground">Delete</Button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// =====================================================
// MAIN COMPARE PAGE
// =====================================================

export default function ComparePage() {
  return (
    <div className="min-h-screen bg-base text-base-color p-4 md:p-8">
      <div className="max-w-4xl mx-auto">
        <div className="text-center mb-6">
          <div className="flex items-center justify-center gap-3 mb-3">
            <div className="w-12 h-12 bg-purple-500/20 rounded-xl flex items-center justify-center text-2xl">⚖️</div>
            <h1 className="text-3xl md:text-4xl font-bold text-base-color">Scales of Justice</h1>
          </div>
          <p className="text-muted-foreground max-w-2xl mx-auto">
            Upload two document images to compare using multi-region forensic analysis.
          </p>
        </div>
        <CompareTab />
      </div>
    </div>
  );
}
