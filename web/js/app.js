(()=>{
const $=id=>document.getElementById(id);
const TOK=new URLSearchParams(location.search).get('token')||'';
const hdr=(h={})=>{if(TOK)h['X-Dub-Token']=TOK;return h};
const url=u=>fetch(u).then(r=>{if(!r.ok)throw Error(r.status);return r.json()});
let mediaGen=0,creating=false,createGen=0,seekPreview=null,edSeq=0,lastPostAt=0;
const errMsg=async(r,d)=>{try{const j=await r.json();return typeof j.detail==='string'?j.detail:JSON.stringify(j.detail)}catch(e){return d+' (HTTP '+r.status+')'}};
const fmt=s=>{s=Math.max(0,s|0);const h=s/3600|0;s%=3600;const m=s/60|0;return(h?h+':'+String(m).padStart(2,'0'):m)+':'+String(s%60).padStart(2,'0')};
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const E=encodeURIComponent,SAFE=/^[A-Za-z0-9_-]{1,32}$/;
const okCode=k=>SAFE.test(String(k))&&!(String(k) in Object.prototype);const NP=()=>Object.create(null);
const num=(x,d)=>{x=Number(x);return Number.isFinite(x)?x:(d||0)},nn=x=>(x==null||x===''||!Number.isFinite(Number(x)))?null:Number(x);
function normSync(y){if(!y||typeof y!=='object')return null;return{segments:num(y.segments),overflowed:num(y.overflowed),cut_s:num(y.cut_s),refit:num(y.refit),tempo_mean:nn(y.tempo_mean),tempo_max:nn(y.tempo_max),tempo_n:num(y.tempo_n),lipsync:y.lipsync===true,chunks_done:num(y.chunks_done),lipsync_chunks:num(y.lipsync_chunks)}}
function normJob(d){
 d=d&&typeof d==='object'?d:{};
 const o={url:typeof d.url==='string'?d.url.slice(0,500):'',job_id:String(d.job_id||''),state:String(d.state||''),queue_position:nn(d.queue_position),queue_len:nn(d.queue_len),error_tail:d.error_tail==null?null:String(d.error_tail),lipsync:d.lipsync!==false,duration:num(d.duration),now:typeof d.now==='string'?d.now:null};
 o.chunks=(Array.isArray(d.chunks)?d.chunks:[]).map(c=>({idx:num(c&&c.idx)|0,start:num(c&&c.start),dur:num(c&&c.dur),has_source:!!(c&&c.has_source)}));
 o.langs=NP();const L=d.langs&&typeof d.langs==='object'?d.langs:{};
 for(const k of Object.keys(L)){if(!okCode(k))continue;const l=L[k]||{};
  o.langs[k]={done:num(l.done),total:l.total==null?null:num(l.total),sync:normSync(l.sync),chunks:(Array.isArray(l.chunks)?l.chunks:[]).map(c=>({idx:num(c&&c.idx)|0,state:String((c&&c.state)||''),stage:c&&c.stage!=null?String(c.stage):'',version:num(c&&c.version)|0,has_media:!!(c&&c.has_media),sync:normSync(c&&c.sync),started:c&&typeof c.started==='string'?c.started:null,finished:c&&typeof c.finished==='string'?c.finished:null,editing:!!(c&&c.editing)}))}}
 return o}
function normSegs(d){return((d&&Array.isArray(d.segments))?d.segments:[]).map(g=>({chunk:num(g.chunk)|0,id:String(g.id),start:num(g.start),end:num(g.end),speaker:num(g.speaker)|0,gender:String(g.gender||''),src_text:g.src_text==null?'':String(g.src_text),tgt_text:g.tgt_text==null?'':String(g.tgt_text),edited:!!g.edited,expr_applied:g.expr_applied===true,ctl_head:Number.isInteger(g.ctl_head_revision)?g.ctl_head_revision:null,fallback:!!g.fallback,fallback_reason:g.fallback_reason==null?null:String(g.fallback_reason),history_len:num(g.history_len),...Object.fromEntries(['dub_start','dub_end','speech_start','speech_end'].filter(k=>g[k]!=null&&isFinite(+g[k])).map(k=>[k,+g[k]]))})).sort((a,b)=>a.start-b.start)}
const CH='#F87171,#34D399,#60A5FA,#FB923C,#C084FC,#FACC15,#2DD4BF,#F472B6,#38BDF8,#A3E635'.split(',');
const col=x=>CH[(x|0)%CH.length];
const GT=g=>g==='female'?'female voice':g==='male'?'male voice':'voice';
const DEMOS=[
 {url:'https://www.youtube.com/watch?v=vTqc2wHT5A0',start:131,dur:180,langs:['hi','ta','bn']},
 {url:'https://www.youtube.com/watch?v=R9IbfB0B4FU',start:1,dur:0,langs:['hi','ta','bn']},
 {url:'https://www.youtube.com/watch?v=vHJSRCkmXk4',start:301,dur:180,langs:['en']}];
let S=null,pollT=null,toastT=null,lastPost='',LANGS=[],picked=[],langErr=false;
const form=$('form'),v=$('v'),pre=$('pre'),ov=$('ov'),ovt=$('ovt'),ovs=$('ovs'),bar=$('bar'),tlabel=$('tlabel'),segbar=$('segbar'),tracksBox=$('tracks'),tlang=$('tlang'),segsBox=$('segs'),toast=$('toast'),pp=$('pp'),stage=$('stage');
const view=n=>{document.body.dataset.view=n;$('landing').hidden=n!=='landing';$('studio').hidden=n!=='studio';$('newbtn').hidden=n!=='studio';$('helpnew').hidden=n!=='studio';FX.view(n)};

const dur=()=>S&&S.data?(S.data.duration||0):0;
const chunks=()=>S&&S.data?(S.data.chunks||[]):[];
function chunkStart(i){const c=chunks();let s=0;for(const x of c){if(x.idx===i)return x.start;s+=x.dur||0}return 0}
function globalTime(){if(S&&S.loading)return S.desired;return(!S||S.chunk==null)?(S&&S.pending||0):(chunkStart(S.chunk)+v.currentTime)}
function chunkIdxOf(t){const c=chunks();if(!c.length)return 0;let acc=0;for(const x of c){if(t<acc+(x.dur||0))return x.idx;acc+=x.dur||0}return c[c.length-1].idx}
function ver(track,i){if(track==='original'||!S.data)return 0;const k=(S.data.langs[track]||{}).chunks;return(k&&k.find(c=>c.idx===i)||{}).version||0}
const lipReady=(t,i)=>{const k=S&&S.data&&S.data.langs&&S.data.langs[t];const c=k&&k.chunks.find(x=>x.idx===i);return!!(c&&c.sync&&c.sync.lipsync===true)};
const lipWanted=()=>!!(S&&S.data&&S.data.lipsync!==false&&S.lipview!=='aligned');
function srcFor(t,i){return t==='original'?`/media/${E(S.job)}/original/${num(i)|0}.mp4`:`/media/${E(S.job)}/${E(t)}/chunks/${num(i)|0}.mp4?v=${ver(t,i)}${lipWanted()?(lipReady(t,i)?'&l=1':''):'&lip=0'}`}
function ready(t,i){if(S&&S.bad&&S.bad[t+':'+i+':'+ver(t,i)])return false;if(t==='original'){const c=chunks().find(x=>x.idx===i);return!!(c&&c.has_source)}const l=S.data&&S.data.langs[t];if(!l)return false;const c=l.chunks&&l.chunks.find(x=>x.idx===i);return!!(c&&c.state==='done'&&c.has_media)}
const chunkInfo=(l,i)=>{const k=((S.data&&S.data.langs[l])||{}).chunks;return k&&k.find(x=>x.idx===i)};
const LN=l=>(S&&S.langName[l])||l;

// ---- adaptive start buffer (unchanged logic from index4) ----
function srvNow(){const n=S.data&&S.data.now?Date.parse(S.data.now):NaN,age=performance.now()-(S.dataAt||0);return(isFinite(n)&&age<20000)?n+age:NaN}
function bufPlan(g,only){
 if(!S||!S.data||S.track==='original')return null;
 const l=S.data.langs[S.track];if(!l||!l.chunks)return null;
 if(g==null)g=globalTime();
 const cs=chunks(),ci=l.chunks,cEnd=i=>chunkStart(i)+((cs.find(x=>x.idx===i)||{}).dur||0);
 const pend=ci.filter(c=>!(c.state==='done'&&c.has_media)&&(only!=null?c.idx===only:chunkStart(c.idx)>g+0.01));
 if(!pend.length)return null;
 const dn=ci.filter(c=>c.state==='done'&&c.has_media&&c.started&&c.finished);
 const st=ci.filter(c=>c.started).map(c=>Date.parse(c.started));
 if(!dn.length||!st.length)return{wait:null};
 const P=Math.min(...st),last=dn.reduce((a,b)=>Date.parse(b.finished)>Date.parse(a.finished)?b:a);
 const fl=Date.parse(last.finished),el=cEnd(last.idx);
 const rate=Math.max(0.05,(fl-P)/1000/Math.max(1,el)),now=srvNow();
 if(!isFinite(P)||!isFinite(fl)||!isFinite(rate)||!isFinite(now))return{wait:null};
 let wait=0,eta1=null,first=1e9;
 for(const c of pend){const R=fl+Math.max(0,cEnd(c.idx)-el)*rate*1000,eta=Math.max(0,(R-now)/1000),buf=(chunkStart(c.idx)-g)/(S.rate||1);wait=Math.max(wait,eta-buf);if(c.idx<first){first=c.idx;eta1=eta}}
 return{wait,rate,eta1};
}
function bufText(idx){const b=bufPlan(chunkStart(idx),idx);return(b&&b.wait!=null)?'Buffering… ~'+Math.max(1,Math.ceil(b.eta1))+'s':'Dubbing this part…'}
// overlay helper: kind = buf | wait | err
function showOv(title,sub,kind){ovt.textContent=title;ovs.textContent=sub||'';ov.dataset.kind=kind;ov.style.display='grid';
 const a=$('ova');a.innerHTML='';
 const mk=(t,f)=>{const b=document.createElement('button');b.textContent=t;b.addEventListener('click',f);a.appendChild(b)};
 if(kind==='buf')mk('Cancel playback',()=>setWant(false));
 if(kind==='langwait')mk('Cancel',()=>cancelWait());
 if(kind==='wait'&&S&&S.track!=='original')mk('Play original',()=>{const g=S.pending!=null?S.pending:globalTime();hideOv();doSwitch('original',g)});
 if(kind==='err'){mk('Try again',()=>{if(!S)return;const own=S,g=globalTime(),trk=S.track,key=S.curKey;dropLoad();playSeq++;mediaGen++;const seq=(S.retrySeq=(S.retrySeq||0)+1);if(key)delete S.bad[key];S.chunk=null;S.curSrc=null;v.removeAttribute('src');S.want=true;pollTick().then(()=>{if(S===own&&S.track===trk&&S.retrySeq===seq)goTo(g)})});if(S.track!=='original')mk('Play original',()=>{hideOv();doSwitch('original',globalTime())});mk('Copy details',()=>copyText(`job ${S.job}: media for ${S.curKey||'this part'} could not be loaded`))}
 renderBanner()}
function hideOv(){ov.style.display='none';delete ov.dataset.kind}
function stopBuffer(){if(S&&S.bufT){clearInterval(S.bufT);S.bufT=null}if(S){S.buffering=false}hideOv();delete ov.dataset.buffering;if(S)renderBanner()}
let playSeq=0;
function playV(){const o=S,gen=mediaGen,src=v.getAttribute('src'),my=++playSeq;
 const fail=e=>{if(e&&e.name==='AbortError')return;if(my!==playSeq||S!==o||gen!==mediaGen||v.getAttribute('src')!==src)return;if(S){S.want=false;if(ov.dataset.kind!=='err')stopBuffer();updatePlayerUI()}showToast('Playback could not start. Press Play to try again.',4000,true)};
 let p;try{p=v.play()}catch(e){fail(e);return}
 if(p&&p.then)p.then(()=>{},fail)}
function startBuffer(){
 S.want=true;S.buffering=true;v.pause();updatePlayerUI();
 const tick=()=>{
  if(!S||!S.want){stopBuffer();return}
  const b=bufPlan();
  if(!b||b.wait==null||b.wait<=0.5){stopBuffer();if(S.chunk!=null){playV();updatePre()}else if(S.pending!=null)goTo(S.pending);return}
  const n=Math.max(1,Math.ceil(b.eta1));
  showOv('Buffering… ~'+n+'s','Starting in about '+n+' seconds. Waiting briefly so playback continues smoothly.','buf');ov.dataset.buffering='1';
 };
 if(S.bufT)clearInterval(S.bufT);S.bufT=setInterval(tick,500);tick();
}

function liveSay(m){$('live-status').textContent='';setTimeout(()=>{$('live-status').textContent=m},30)}
function showToast(m,ms,err){if(err)ms=Math.max(ms||0,8000);toast.textContent=m;toast.classList.toggle('e',!!err);toast.classList.add('on');clearTimeout(toastT);toastT=setTimeout(()=>toast.classList.remove('on'),ms||3000);if(err){$('live-error').textContent='';setTimeout(()=>{$('live-error').textContent=m},30)}else liveSay(m)}

// ---- part state model ----
const stepName=s=>{s=String(s||'').toLowerCase().replace(/[\s-]+/g,'_');if(['','start','starting','fetch','audio','download','source','extract','prepare','ingest'].includes(s))return'Getting video';if(['vad','slots','speakers','asr','diarize','diarization','transcribe','speech'].includes(s))return'Finding speech and speakers';if(['mt','fit','translate','translation','refit'].includes(s))return'Translating';if(['tts','voice','synthesize','synthesis'].includes(s))return'Creating voices';return'Mixing video'};
function partState(l,i){
 const c=chunkInfo(l,i);if(!c)return{k:'wait',t:'Waiting',ic:'○'};
 if(S.bad&&S.bad[l+':'+i+':'+ver(l,i)])return{k:'fail',t:'Could not load',ic:'!'};
 if(c.editing)return{k:'edit',t:'Updating',ic:'✎'};
 if(c.state==='error')return{k:'fail',t:'Failed',ic:'!'};
 if(c.state==='done'&&c.has_media)return(c.sync&&c.sync.lipsync===true)?{k:'lips',t:'Lips matched',ic:'✓'}:{k:'voice',t:(S&&S.data&&S.data.lipsync===false)?'Aligned audio':'Voice ready',ic:'●'};
 if(c.started||c.state==='running')return{k:'work',t:'Working',sub:stepName(c.stage),ic:'◐'};
 return{k:'wait',t:'Waiting',ic:'○'};
}

// ---- banner ----
function renderBanner(){renderBanner0();const b=$('banner');document.documentElement.style.setProperty('--bh',b.hidden?'0px':b.offsetHeight+'px')}
function renderBanner0(){
 const b=$('banner');if(!S||!S.job){b.hidden=true;return}
 let k='info',h='',d='',acts=[],spin=false;const st=S.data&&S.data.state,c=chunks();
 if(S.conn){k='bad';h='Connection lost';d='Trying to reconnect. Playback of loaded media may continue.';acts=[['Retry now',()=>pollTick()]]}
 else if(S.data&&S.data.state==='error'){k='bad';h='This dubbing job stopped';d='Some ready parts may still play.';acts=[['Retry status',()=>{pollFails=0;startPoll();pollTick()}],['Copy details',()=>copyText(`job ${S.job}, state error: ${S.data.error_tail||'no details'}`)],['New dub',()=>resetStudio()]]}
 else if(S.buffering&&ov.dataset.kind==='buf'){k='info';spin=true;h=ovt.textContent.replace('Buffering… ~','Starting in about ').replace(/s$/,' seconds');d='Waiting briefly so playback can continue smoothly.';acts=[['Cancel playback',()=>setWant(false)]]}
 else if(ov.style.display==='grid'&&ov.dataset.kind==='wait'){k='info';spin=true;h='Dubbing this part';d='Playback will continue when this part is ready.'}
 else if(!S.data){k='info';spin=true;h='Preparing the video';d='Getting the video and dividing it into playable parts.'}
 else if(st==='queued'){k='warn';spin=true;const q=S.data.queue_position,n=S.data.queue_len;h=(q&&n)?`Waiting in line: ${q} of ${n}`:'Waiting to start';d=(q&&n)?'One new video is prepared at a time. You can leave this page open.':'Your video will begin as soon as the current job finishes.'}
 else if(st==='planning'){spin=true;h='Preparing the video';d='Getting the video and dividing it into playable parts.'}
 else if(st==='done'&&!c.length){k='warn';h='No playable parts';d='The video finished processing but no playable parts were produced.'}
 else if(st==='done'){if(S.dismissed){b.hidden=true;return}k='ok';h=S.requestedLangs.map(LN).join(', ')+' ready';d=`${S.requestedLangs.length} language${S.requestedLangs.length===1?'':'s'} · ${c.length} parts`;acts=[['Dismiss',()=>{S.dismissed=true;renderBanner()}]]}
 else{let tot=0,dn=0;S.requestedLangs.forEach(l=>{const o=S.data.langs[l]||{};tot+=o.total!=null?o.total:c.length;dn+=o.done||0});
  const first=S.requestedLangs.some(l=>ready(l,0));
  if(!first){spin=true;h='Creating the first dubbed part';const w=S.requestedLangs.map(l=>chunkInfo(l,0)).find(x=>x&&(x.started||x.state==='running'));d=w?stepName(w.stage)+'…':'You can leave this page open while the first part is created.'}
  else{spin=true;h=`${dn} of ${tot} parts ready`;d='You can watch ready parts while the rest continue.'}}
 b.dataset.k=k;b.hidden=false;$('bt1').innerHTML=(spin?'<span class=spin aria-hidden=true></span> ':'')+esc(h);$('bt2').textContent=d;
 const a=$('bact');a.innerHTML='';acts.forEach(([t,f])=>{const x=document.createElement('button');x.textContent=t;x.addEventListener('click',f);a.appendChild(x)});
 const key=k+h;if(S.lastBanner!==key){S.lastBanner=key;if(k==='bad'||k==='ok'||st==='queued'){liveSay(h)}}
}
async function copyText(t){try{await navigator.clipboard.writeText(t);showToast('Details copied.')}catch(e){showToast('Could not copy details.',3000,true)}}

// ---- landing: languages ----
function renderPick(){
 const q=$('psearch').value.trim().toLowerCase(),lst=$('picklist');
 if(langErr){lst.innerHTML='<div class=empty>Languages could not be loaded. <button type=button id=lretry>Retry</button></div>';$('lretry').onclick=loadLangs;return}
 if(!LANGS.length){lst.innerHTML=LANGS_LOADED?'<div class=empty>No target languages are available right now.</div>':'<div aria-hidden=true>'+'<div class=sk></div>'.repeat(6)+'</div><div class=help>Loading languages…</div>';return}
 const m=LANGS.filter(l=>!q||(l.name+' '+(l.script||'')+' '+l.code).toLowerCase().includes(q));
 const full=picked.length>=3;
 const row=l=>{const on=picked.includes(l.code),dis=!on&&full;return `<label class="opt${dis?' dis':''}"><input type=checkbox value="${esc(l.code)}" ${on?'checked':''} ${dis?'disabled':''}><span>${esc(l.name)}</span>${l.status==='polished'?'<span class="pill ok">Polished</span>':'<span class="pill warn">Preview</span>'}<span class=s>${esc(l.script||'')}</span></label>`};
 const pol=m.filter(l=>l.status==='polished'),oth=m.filter(l=>l.status!=='polished');
 lst.innerHTML=!m.length?`<div class=empty>No languages match “${esc($('psearch').value.trim())}”.</div>`:(pol.length?'<div class=lh>Polished</div>'+pol.map(row).join(''):'')+(oth.length?'<div class=lh>Preview languages</div>'+oth.map(row).join(''):'');
 $('pickhelp').textContent=full?'You can choose up to three languages. Remove one to choose another.':'';
 const names=picked.map(c=>(LANGS.find(l=>l.code===c)||{name:c}).name);
 $('picktxt').textContent=!names.length?'Choose up to 3 languages':names.length<=2?names.join(', '):names.slice(0,2).join(', ')+' +'+(names.length-2);
 $('pickn').textContent=picked.length+' of 3 selected';
}
let LANGS_LOADED=false;
async function loadLangs(){langErr=false;LANGS_LOADED=false;renderPick();updateSubmit();
 try{LANGS=(await url('/api/languages')).targets||[];LANGS=LANGS.filter(l=>l&&okCode(l.code))}catch(e){langErr=true}
 LANGS_LOADED=true;
 renderPick();updateSubmit();if(S)S.languages=LANGS;LANGS.forEach(l=>{if(S)S.langName[l.code]=String(l.name||l.code)});
 if(S&&S.data&&S.requestedLangs.length){fillTlang();renderProg()}}
function fillTlang(){const cur=S.lang;tlang.innerHTML=S.requestedLangs.map(l=>`<option value="${esc(l)}">${esc(LN(l))}</option>`).join('');if(cur&&cur!=='original')tlang.value=cur}

// ---- time parsing ----
function parseT(s){s=String(s).trim();if(!s)return NaN;if(/^\d+(\.\d+)?$/.test(s))return parseFloat(s);const p=s.split(':');if(p.length>3||p.some(x=>!/^\d+(\.\d+)?$/.test(x)))return NaN;if(p.slice(1).some(x=>parseFloat(x)>=60))return NaN;return p.reduce((a,x)=>a*60+parseFloat(x),0)}
function ytOk(raw){try{const u=new URL(raw);if(!/^https?:$/.test(u.protocol))return false;const h=u.hostname.toLowerCase().replace(/^www\./,'');return['youtube.com','m.youtube.com','music.youtube.com','youtu.be','youtube-nocookie.com'].includes(h)}catch(e){return false}}
function rangeCheck(){const whole=document.querySelector('input[name=rng]:checked').value==='whole';if(whole)return{ok:true,start:0,dur:0};
 const sv=$('start').value.trim(),ev=$('end').value.trim();if(!sv)return{ok:false,msg:'Enter a start time.'};if(!ev)return{ok:false,msg:'Enter an end time.'};
 const s=parseT(sv),en=parseT(ev);if(!isFinite(s))return{ok:false,msg:'Enter a valid start time.'};if(!isFinite(en))return{ok:false,msg:'Enter a valid end time.'};
 if(en===s)return{ok:false,msg:'Start and end are the same time.'};if(en<s)return{ok:false,msg:'End time must be after start time.'};return{ok:true,start:s,dur:en-s}}
function formOk(){return!creating&&LANGS_LOADED&&!langErr&&ytOk($('url').value.trim())&&picked.length>=1&&picked.length<=3&&rangeCheck().ok&&$('consent').checked}
function updateSubmit(){const b=$('dub');if(b)b.disabled=!formOk();const r=rangeCheck();const cl=$('cliplen');if(cl)cl.textContent=r.ok&&r.dur?'Clip length: '+fmt(r.dur)+'.':''}
function fe(id,msg){const e=$(id);e.hidden=!msg;e.textContent=msg||''}

function newState(j,ck){
 FX.reset();lastPost='';lastPostAt=0;seekPreview=null;
 S={job:String(j.job_id),data:null,track:'original',lang:ck[0],chunk:null,want:false,pending:null,languages:LANGS,requestedLangs:(Array.isArray(j.langs)?j.langs:ck).filter(l=>okCode(l)),langName:NP(),transcripts:NP(),tseq:NP(),tErr:NP(),tStale:NP(),bad:NP(),prev:null,follow:true,mut:false,ended:false,loadH:null,curSrc:null,curKey:null,captions:S&&S.captions||false,conn:false,dismissed:false,rate:1};
 LANGS.forEach(l=>S.langName[l.code]=String(l.name||l.code));
}
async function postJob(payload,errId){
 const r=await fetch('/api/jobs',{method:'POST',headers:hdr({'Content-Type':'application/json'}),body:JSON.stringify(payload)});
 if(!r.ok)throw Error(await errMsg(r,'error'));return r.json();
}
function enterStudio(j,ck){
 if(!/^\w+$/.test(String(j&&j.job_id||'')))throw Error('The server returned an invalid job id.');
 abortPoll();if(S&&S.bufT)clearInterval(S.bufT);dropLoad();mediaGen++;try{v.pause();v.removeAttribute('src')}catch(e){}
 newState(j,ck);pollFails=0;location.hash='job='+j.job_id;
 applyView();startPoll();pollTick();loadTranscript();
}
function setCreating(b){creating=b;document.querySelectorAll('.demo').forEach(c=>c.disabled=b);updateSubmit()}
async function createJob(e){
 if(e)e.preventDefault();if(creating)return;
 const raw=$('url').value.trim();
 fe('urle','');fe('lange','');fe('rnge','');fe('conse','');$('url').removeAttribute('aria-invalid');
 if(!ytOk(raw)){fe('urle',raw?'Enter a valid YouTube link.':'Enter a YouTube video URL.');$('url').setAttribute('aria-invalid','true');return}
 if(!picked.length){fe('lange','Choose at least one language.');return}
 const rc=rangeCheck();if(!rc.ok){fe('rnge',rc.msg);return}
 if(!$('consent').checked){fe('conse','Confirm that you have permission before starting.');return}
 $('formmsg').hidden=true;$('dub').textContent='Starting…';const my=++createGen;setCreating(true);
 try{const ck=picked.slice(),j=await postJob({url:raw,langs:ck,start:rc.start,dur:rc.dur,lipsync:$('lipopt').checked});if(my!==createGen)return;enterStudio(j,ck)}
 catch(x){if(my===createGen){$('formmsg').hidden=false;$('formmsg').textContent='Could not start this dubbing job. '+String(x.message||x).slice(0,200)}}
 finally{if(my===createGen){setCreating(false)}$('dub').textContent='Start dubbing'}
}
async function openDemo(n){
 if(creating)return;
 const p=DEMOS[n],cards=[...document.querySelectorAll('.demo')],my=++createGen;setCreating(true);
 const card=cards[n],old=card.querySelector('.rd').textContent;card.querySelector('.rd').textContent='Opening demo…';$('demoerr').hidden=true;
 try{const j=await postJob(p);if(my!==createGen)return;enterStudio(j,p.langs)}
 catch(x){if(my===createGen){$('demoerr').hidden=false;$('demoerr').textContent='The demo could not be opened. Try again.'}}
 finally{if(my===createGen)setCreating(false);card.querySelector('.rd').textContent=old}
}

// ---- rendering ----
function renderProg(){
 if(!S.data)return;syncAddedLangs();renderBanner();
 const st=S.data.state,c=chunks();
 if((st==='planning'||st==='queued')&&!c.length){$('parts').innerHTML='<div class=empty>Parts will appear after the video is prepared.</div>';renderSync();return}
 renderTracks();renderSegbar();renderParts();renderSync();updatePlayerUI();
 FX.meta({url:S.data.url,job:S.job,duration:S.data.duration,parts:c.length,state:st,langs:S.requestedLangs.map(LN)});
 const edc=S.track!=='original'&&S.chunk!=null&&(chunkInfo(S.track,S.chunk)||{}).editing;$('editing-tag').hidden=!edc;
}
function renderTracks(){
 if(!S.data)return;
 const mk=(code,label)=>{const act=S.track===code;let n='',dot='',al=label;
  if(code!=='original'){const o=S.data.langs[code]||{},t=o.total!=null?o.total:chunks().length;n=`<span class=n>${o.done||0}/${t}</span>`;al=label+', '+(o.done||0)+' of '+t+' parts ready';
   const idx=chunkIdxOf(globalTime()),ps=partState(code,idx);dot=`<span class="dot ${ps.k==='lips'?'lips':ps.k==='voice'?'voice':ps.k==='work'?'work':''}" aria-hidden=true></span>`}
  return `<button class=tt data-trk="${esc(code)}" aria-label="${esc(al)}" aria-pressed="${act}">${dot}${esc(label)}${n}</button>`};
 setHtml(tracksBox,mk('original','Original')+S.requestedLangs.map(l=>mk(l,LN(l))).join(''));FX.tracks();
 const diff=S.lang&&S.track!=='original'&&tlang.value&&tlang.value!==S.track;
 $('ctx').textContent=diff?`Showing ${LN(tlang.value)} transcript · Playing ${LN(S.track)}`:'';
}
function setHtml(el,h){if(el.__h===h)return;el.__h=h;const a=document.activeElement;let kind=null,k1=null,k2=null;
 if(a&&el.contains(a)){if(a.dataset.trk!=null){kind='t';k1=a.dataset.trk}else if(a.dataset.part!=null){kind='p';k1=a.dataset.lang;k2=a.dataset.part}}
 el.innerHTML=h;
 if(kind){const n=kind==='t'?el.querySelector('.tt[data-trk="'+CSS.escape(k1)+'"]'):el.querySelector('.pc[data-lang="'+CSS.escape(k1)+'"][data-part="'+CSS.escape(k2)+'"]');if(n)n.focus({preventScroll:true})}}
function renderSegbar(){
 const c=chunks();if(!c.length){segbar.innerHTML='';return}
 const t=S.track;
 segbar.innerHTML=c.map(x=>{let cl='';if(t==='original')cl=ready(t,x.idx)?'o':'';else{const ps=partState(t,x.idx);cl=ps.k==='lips'?'l':ps.k==='voice'?'v':ps.k==='fail'?'f':''}
  if(S.chunk===x.idx)cl+=' cur';return `<span class="${cl}" style="flex:${x.dur||1} 1 0"></span>`}).join('');
}
function renderParts(){
 const c=chunks();if(!c.length){$('parts').innerHTML='<div class=empty>Parts will appear after the video is prepared.</div>';return}
 const prev=S.prev,now={};
 setHtml($('parts'),S.requestedLangs.map(l=>`<div class=prow><div class=pl>${esc(LN(l))}</div><div class=cells>`+c.map(x=>{
  const ps=partState(l,x.idx),key=l+':'+x.idx;now[key]=ps.k;
  const up=prev&&prev[key]&&prev[key]!==ps.k&&(ps.k==='voice'||ps.k==='lips');
  if(up&&S.notified!==key+ps.k){S.notified=key+ps.k;
   if(ps.k==='lips'&&prev[key]==='voice')showToast(`Lip matching is ready for ${LN(l)}, Part ${x.idx+1}.`,5000);else liveSay(`${LN(l)}, Part ${x.idx+1}: ${ps.k==='lips'?'lips matched':'voice ready'}`)}
  const cur=l===S.track&&S.chunk===x.idx;
  return `<button class="pc ${ps.k}${up?' pulse':''}${cur?' cur':''}" style="flex:${Math.max(.5,x.dur||1)} 1 0" data-lang="${esc(l)}" data-part="${x.idx}"${cur?' aria-current="true"':''} aria-label="${esc(LN(l))}, Part ${x.idx+1}, ${fmt(x.start)} to ${fmt(x.start+x.dur)}, ${ps.t}${ps.sub?' · '+esc(ps.sub):''}" title="${esc(ps.t)}"><b><i class=pi aria-hidden=true></i>Part ${x.idx+1}</b><span>${ps.t}${ps.sub?' · '+esc(ps.sub):''}</span></button>`}).join('')+'</div></div>').join(''));
 S.prev=now;
}
function qLines(y){
 const L=[];if(!y||!y.segments)return L;
 L.push(y.overflowed===0&&y.cut_s===0?['ok','Nothing was truncated.']:['warn',`${y.overflowed} of ${y.segments} lines exceeded their available time; ${y.cut_s.toFixed(1)} seconds were shortened.`]);
 L.push(y.refit===0?['ok','No lines needed automatic shortening.']:['info',`${y.refit} line${y.refit===1?' was':'s were'} automatically shortened to fit.`]);
 if(y.tempo_mean==null||y.tempo_max==null)L.push(['info','Speech-speed measurements are not ready yet.']);
 else{const m=y.tempo_mean.toFixed(2),x=y.tempo_max.toFixed(2);
  L.push((y.tempo_mean>1.10||y.tempo_max>1.20)?['warn',`Some speech was sped up to fit: ${m}× average, ${x}× fastest.`]:['ok',`Speech stayed close to natural speed: ${m}× average, ${x}× fastest.`])}
 if(S&&S.data&&S.data.lipsync===false)L.push(['info','Lip sync is off for this job; the dubbed audio is aligned to the original video.']);else L.push(y.chunks_done>0?(y.lipsync_chunks===y.chunks_done?['ok',`Lips are matched in all ${y.chunks_done} ready parts.`]:y.lipsync_chunks>0?['info',`Lips are matched in ${y.lipsync_chunks} of ${y.chunks_done} ready parts.`]:['info','Voice audio is ready; lip matching is not yet available for these parts.']):['info','Lip matching will be reported after the first part is ready.']);
 return L;
}
function renderSync(){
 const b=$('syncb');if(!S||!S.data||!b)return;
 const ys=S.requestedLangs.map(l=>(S.data.langs[l]||{}).sync);
 if(!ys.some(y=>y&&y.segments))$('qsum').innerHTML='<div class=empty style="grid-column:1/-1">Quality measurements will appear after the first dubbed part.</div>';
 else $('qsum').innerHTML=S.requestedLangs.map((l,i)=>{const y=ys[i];return `<div class=qc><h3>${esc(LN(l))}</h3>`+(y&&y.segments?'<ul>'+qLines(y).map(([k,t])=>`<li class=${k}><i aria-hidden=true>${k==='ok'?'✓':k==='warn'?'!':'i'}</i><span>${esc(t)}</span></li>`).join('')+'</ul>':'<p class=mut>Waiting for the first part…</p>')+'</div>'}).join('');
 b.innerHTML=S.requestedLangs.map((l,i)=>{const y=ys[i];
  if(!y||!y.segments)return `<tr data-lang="${esc(l)}"><td>${esc(LN(l))}</td><td class=sy-seg>—</td><td class=sy-ovf>—</td><td class=sy-cut>—</td><td class=sy-refit>—</td><td class=sy-tempo>—</td><td class=sy-ls>—</td></tr>`;
  const ocls=y.overflowed?'warn':'okc',ccls=y.cut_s>0?'warn':'okc';
  return `<tr data-lang="${esc(l)}"><td>${esc(LN(l))}</td><td class=sy-seg>${y.segments}</td><td class="sy-ovf ${ocls}">${y.overflowed} (${(100*y.overflowed/y.segments).toFixed(0)}%)</td><td class="sy-cut ${ccls}">${y.cut_s.toFixed(1)}</td><td class=sy-refit>${y.refit}</td><td class=sy-tempo>${y.tempo_mean==null?'–':y.tempo_mean.toFixed(2)} / ${y.tempo_max==null?'–':y.tempo_max.toFixed(2)}</td><td class=sy-ls>${y.lipsync_chunks} / ${y.chunks_done}</td></tr>`}).join('');
}
function lipAvail(){const L=S&&S.data&&S.data.langs&&S.data.langs[S.track];return!!(L&&L.chunks.some(c=>c.state==='done'&&c.has_media&&c.sync&&c.sync.lipsync===true))}
function lipBtnSync(){const b=$('lipbtn');if(!b)return;const show=!!(S&&S.data&&S.data.lipsync!==false&&S.track!=='original'&&lipAvail());b.hidden=!show;if(!show)return;const on=lipWanted();b.setAttribute('aria-pressed',String(on));b.setAttribute('aria-label',on?'Lip sync: on. Switch to the original video':'Original video. Switch to lip sync');b.title=on?'Showing lip-synced video (click for original video)':'Showing original video with aligned audio (click for lip sync)'}
function lipToggle(){if(!S||!S.data||S.data.lipsync===false||!lipAvail())return;S.lipview=lipWanted()?'aligned':'lip';lipBtnSync();showToast(lipWanted()?'Lip-synced video.':'Original video with aligned audio.',2200);if(S.track==='original'||S.chunk==null)return;const gt=globalTime();S.chunk=null;S.curSrc=null;v.removeAttribute('src');goTo(gt)}
function updatePlayerUI(){lipBtnSync();if(S&&S.wait)tickWait();
 if(!S)return;const d=dur(),g=S.ended?d:globalTime(),none=!!(S.data&&!chunks().length);
 if(seekPreview===null){bar.max=d||100;bar.value=Math.min(g,d||g);tlabel.textContent=fmt(g)+' / '+fmt(d);$('fill').style.width=(d?Math.min(100,g/d*100):0)+'%'}
 pp.innerHTML=FX.ic(S.want?'pause':'play');pp.setAttribute('aria-label',S.want?'Pause':'Play');$('bigplay').setAttribute('aria-label','Play');
 pp.disabled=none;$('bigplay').disabled=none;$('mpp').disabled=none;$('mpp').innerHTML=FX.ic(S.want?'pause':'play');$('mpp').setAttribute('aria-label',S.want?'Pause':'Play');$('mtime').textContent=fmt(g)+' / '+fmt(d);$('mcc').setAttribute('aria-pressed',String(!!S.captions));
 stage.classList.toggle('idle',!S.want);
 const idx=chunkIdxOf(g),ps=S.track==='original'?{t:ready('original',idx)?'Original':'Not ready'}:partState(S.track,idx);
 bar.setAttribute('aria-valuetext',fmt(g)+' of '+fmt(d)+', Part '+(idx+1)+', '+ps.t);
 $('mstat').textContent=S.track==='original'?('Original'+(FX.src?' ('+FX.src+')':'')):(LN(S.track)+' · '+ps.t);$('mstat').dataset.k=S.track==='original'?'':(ps.k||'');
}
function highlight(){
 const g=globalTime();let best=null;const rows=segsBox.querySelectorAll('.seg-row');for(const r of rows){if(+r.dataset.start<=g+0.01)best=r;else break}
 if(best&&g>+best.dataset.end+1.5)best=null;
 const cur=segsBox.querySelector('.seg-row.cur');
 if(best!==cur){if(cur){cur.classList.remove('cur');cur.removeAttribute('aria-current')}if(best){best.classList.add('cur');best.setAttribute('aria-current','true');followScroll(best)}}
 updateCaption();
}
function followScroll(row){
 if(!S||!S.follow||segsBox.querySelector('.ed'))return;
 const b=segsBox,top=row.offsetTop-b.offsetTop,h=row.offsetHeight;
 if(top<b.scrollTop+20||top+h>b.scrollTop+b.clientHeight-20){S.prog=true;b.scrollTop=Math.max(0,top-b.clientHeight/2+h/2);setTimeout(()=>{if(S)S.prog=false},80)}
}
/* ---- /M-4 subtitles: cue builder (pure), renderer, layout ---- */
const SUB=(function(){
 const MAXG=32,MINMS=1000,MAXMS=6000,STRONG=/[.!?।॥؟۔…]$/,WEAK=/[,;:،؛]$/;
 let segG=null;try{segG=new Intl.Segmenter('und',{granularity:'grapheme'})}catch(e){}
 const gr=s=>{s=String(s==null?'':s);return segG?Array.from(segG.segment(s),x=>x.segment):Array.from(s)};
 const glen=s=>gr(s).length;
 function script(s){const c={dev:0,tam:0,ara:0,lat:0,oth:0};for(const ch of String(s)){const p=ch.codePointAt(0);if(p>=0x900&&p<=0x97F)c.dev++;else if(p>=0xB80&&p<=0xBFF)c.tam++;else if((p>=0x600&&p<=0x6FF)||(p>=0x750&&p<=0x77F)||(p>=0xFB50&&p<=0xFDFF)||(p>=0xFE70&&p<=0xFEFF))c.ara++;else if(/[A-Za-z]/.test(ch))c.lat++;else if(/\p{L}/u.test(ch))c.oth++}
  let k='oth',m=0;for(const x in c)if(c[x]>m){m=c[x];k=x}return k}
 const CPS={lat:17,dev:14,tam:12,ara:14,oth:13}; /* provisional graphemes/second thresholds per script */
 const RTLL=new Set(['ur','ar','sd','ks','fa','ps','he']),isRtl=(text,lang)=>RTLL.has(String(lang||'').toLowerCase().split(/[-_]/)[0])||script(text)==='ara';
 function words(t){const out=[];for(const w of String(t==null?'':t).replace(/\s+/g,' ').trim().split(' ')){if(!w)continue;const g=gr(w);if(g.length<=MAXG)out.push(w);else for(let i=0;i<g.length;i+=MAXG-1)out.push(g.slice(i,i+MAXG-1).join(''))}return out}
 function fitLines(ws,measure,maxPx){
  const ok=s=>glen(s)<=MAXG&&(!measure||measure(s)<=maxPx),all=ws.join(' ');
  if(ok(all))return[all];
  let best=null,bs=1e9;
  for(let k=1;k<ws.length;k++){const a=ws.slice(0,k).join(' '),b=ws.slice(k).join(' ');if(!ok(a)||!ok(b))continue;
   const wa=measure?measure(a):glen(a),wb=measure?measure(b):glen(b);let s=Math.abs(wa-wb)/Math.max(wa,wb,1);
   if(STRONG.test(ws[k-1]))s-=.35;else if(WEAK.test(ws[k-1]))s-=.2;
   if(/^(and|but|or|that|which|because|so|और|तो|लेकिन|कि|क्योंकि|மற்றும்|ஆனால்|ஏனெனில்|و|لیکن|کہ)$/i.test(ws[k]))s-=.1;
   if(glen(b)<=3)s+=.3;if(glen(a)<=2)s+=.2;
   if(s<bs){bs=s;best=[a,b]}}
  return best}
 function pack(ws,measure,maxPx){const out=[];let i=0;
  while(i<ws.length){let j=i+1,L=fitLines(ws.slice(i,j),measure,maxPx);if(!L){out.push({ws:ws.slice(i,j),lines:[ws[i]]});i=j;continue}
   while(j<ws.length){const l=fitLines(ws.slice(i,j+1),measure,maxPx);if(!l)break;j++;L=l}
   if(j<ws.length){const tot=glen(ws.slice(i,j).join(' '));let done=false;
    for(let k=j;k>i+1&&!done;k--)if(STRONG.test(ws[k-1])&&glen(ws.slice(i,k).join(' '))>=.4*tot){j=k;done=true}
    for(let k=j;k>i+1&&!done;k--)if(WEAK.test(ws[k-1])&&glen(ws.slice(i,k).join(' '))>=.5*tot){j=k;done=true}
    if(done)L=fitLines(ws.slice(i,j),measure,maxPx)||L}
   out.push({ws:ws.slice(i,j),lines:L});i=j}
  return out}
 function hash(s){let h=2166136261;for(let i=0;i<s.length;i++){h^=s.charCodeAt(i);h=Math.imul(h,16777619)}return(h>>>0).toString(36)}
 const nsl=s=>glen(String(s).replace(/\s/g,''));
 /* cues for ONE segment. seg: {id,start,end,src_text,tgt_text,speaker,edited,dub_start?,dub_end?}; times in seconds.
    kind 'dub' uses tgt_text and measured dub onset/end when the API returns them (else the slot; marked "slot"); kind 'orig' uses src_text. */
 function segCues(seg,o){o=o||{};const nextSame=!!o.nextSame,kind=o.kind||'dub',lang=o.lang||'und',measure=o.measure||null,maxPx=o.maxPx||1e9,nextStartMs=o.nextStartMs==null?Infinity:o.nextStartMs;
  const text=kind==='orig'?seg.src_text:seg.tgt_text,ws=words(text);if(!ws.length)return[];
  let a=seg.start,b=seg.end,q='slot';
  if(kind==='dub'){const ds=[seg.dub_start,seg.speech_start,seg.onset].find(x=>typeof x==='number'),de=[seg.dub_end,seg.speech_end,seg.offset].find(x=>typeof x==='number');if(ds!=null){a=ds;q='measured'}if(de!=null){b=de;q='measured'}}
  else q='measured';
  let s0=Math.round(a*1000),e0=Math.max(s0+200,Math.round(b*1000));
  let pcs=pack(ws,measure,maxPx).map(p=>({ws:p.ws,lines:p.lines}));
  const span=e0-s0;
  const alloc=ps=>{const tot=ps.reduce((x,p)=>x+Math.max(1,nsl(p.ws.join(' '))),0);let t=s0;ps.forEach((p,i)=>{const d=i===ps.length-1?e0-t:Math.round(span*Math.max(1,nsl(p.ws.join(' ')))/tot);p.s=t;p.e=t+d;t=p.e})};
  alloc(pcs);
  /* merge pieces shorter than the 1 s minimum into a neighbour when they still fit two lines */
  for(let guard=0;guard<20&&pcs.length>1;guard++){const i=pcs.findIndex(p=>p.e-p.s<MINMS);if(i<0)break;const j=i>0?i-1:i+1,lo=Math.min(i,j),hi=Math.max(i,j),w=pcs[lo].ws.concat(pcs[hi].ws),L=fitLines(w,measure,maxPx);if(!L)break;pcs.splice(lo,2,{ws:w,lines:L});alloc(pcs)}
  /* cues longer than 6 s: split on a clause boundary near the middle, else cap */
  for(let guard=0;guard<20;guard++){const i=pcs.findIndex(p=>p.e-p.s>MAXMS&&p.ws.length>1);if(i<0)break;const p=pcs[i];let k=-1,bd=1e9;for(let x=1;x<p.ws.length;x++)if(STRONG.test(p.ws[x-1])||WEAK.test(p.ws[x-1])){const d=Math.abs(x-p.ws.length/2);if(d<bd){bd=d;k=x}}
   if(k<0)k=Math.round(p.ws.length/2);const w1=p.ws.slice(0,k),w2=p.ws.slice(k),L1=fitLines(w1,measure,maxPx),L2=fitLines(w2,measure,maxPx);if(!L1||!L2)break;
   const mid=p.s+Math.round((p.e-p.s)*nsl(w1.join(' '))/Math.max(1,nsl(p.ws.join(' '))));pcs.splice(i,1,{ws:w1,lines:L1,s:p.s,e:mid},{ws:w2,lines:L2,s:mid,e:p.e})}
  pcs.forEach(p=>{if(p.e-p.s>MAXMS)p.e=p.s+MAXMS});
  const cps=CPS[script(ws.join(' '))]||13;
  const rev=hash((text||'')+(seg.edited?'e':''));
  return pcs.map((p,i)=>{let s=p.s,e=p.e;const last=i===pcs.length-1;
   const txt=p.ws.join(' '),n=nsl(txt);
   /* reading-speed relief: a cue may hold into the silence after it (never past the next cue/speaker) */
   if(last){const need=Math.max(MINMS,Math.ceil(n/cps*1000)),room=Math.min(600,Math.max(0,nextStartMs-e-40));if(e-s<need&&room>0)e=Math.min(e+room,s+need)}
   if(e-s<MINMS){const lim=last?(nextSame?Infinity:nextStartMs-40):pcs[i+1].s;e=Math.max(e,Math.min(s+MINMS,lim))}
   const dur=Math.max(1,e-s),fast=n/(dur/1000)>cps+1e-9;
   return{id:kind+':'+seg.id+':'+i,revision:rev,start_ms:s,end_ms:e,lines:p.lines,lang:lang,timing_quality:pcs.length>1?'approx':q,fast:fast,kind:kind,speaker:seg.speaker,seg:seg.id,cps:Math.round(n/(dur/1000)*10)/10,limit:cps}})}
 return{gr:gr,glen:glen,words:words,fitLines:fitLines,pack:pack,script:script,isRtl:isRtl,segCues:segCues,CPS:CPS,MAXG:MAXG,hash:hash,
  build:(segs,o)=>{const out=[];(segs||[]).forEach((g,i)=>{const nx=segs[i+1];out.push(...segCues(g,Object.assign({},o,{nextStartMs:nx?Math.round(nx.start*1000):Infinity,nextSame:!!nx&&nx.speaker===g.speaker})))});return out}}})();
window.SUB=SUB;
const capIdx={},capArr={};let capMeasure=null,capFont='',capCache=new Map(),capSig='',capGeo={px:320,fs:20};
function capMeasureFn(){const pr=$('capprobe'),cs=getComputedStyle(pr),font=cs.fontStyle+' '+cs.fontWeight+' '+cs.fontSize+' '+cs.fontFamily;if(font!==capFont){capFont=font;capCache.clear();const cv=document.createElement('canvas').getContext('2d');cv.font=font;capMeasure=s=>cv.measureText(s).width}return capMeasure}
function capSubMode(){return!S||!S.captions?'off':(S.subMode||'dub')}
function capLayout(){
 const c=$('cap'),sr=stage.getBoundingClientRect();if(!sr.width)return;
 const sheet=S&&S.sheet||{},narrow=sr.width<=560,gut=Math.max(12,Math.min(120,Math.round(sr.width*(narrow?.04:.08)))),sw=Math.min(sheet.width||0,sr.width*.6);
 const ctlHidden=stage.classList.contains('hidectl');let top=sr.bottom;
 if(!ctlHidden){for(const id of['deck','seekw','tlabel','capbtn','fsbtn','mutebtn']){const e=$(id);if(!e)continue;const r=e.getBoundingClientRect();if(r.width&&r.height&&r.bottom>sr.top+sr.height*.55&&getComputedStyle(e).visibility!=='hidden')top=Math.min(top,r.top)}}
 const safe=parseFloat(getComputedStyle($('capsafe')).paddingBottom)||0;
 let bottom=(sr.bottom-top)+(ctlHidden?Math.max(14,safe+10):10);if(sheet.height)bottom=Math.max(bottom,sheet.height+10);
 c.style.setProperty('--capb',Math.round(bottom)+'px');c.style.left=gut+'px';c.style.right=(gut+sw)+'px';
 const px=Math.max(120,sr.width-2*gut-sw-28),fs=narrow?15:20;
 if(Math.abs(px-capGeo.px)>1||fs!==capGeo.fs){capGeo={px:px,fs:fs};capCache.clear();capSig=''}
 c.style.setProperty('--capfs',fs+'px');$('capprobe').style.fontSize=fs+'px'}
function capCues(arr,kind,tr,g){
 if(!arr||!arr.length)return null;let bi=capIdx[kind];if(capArr[kind]!==arr||bi==null||bi<0||bi>=arr.length||arr[bi].start>g+0.01||(bi+1<arr.length&&arr[bi+1].start<=g+0.01)){let lo=0,hi=arr.length-1;bi=-1;while(lo<=hi){const m=(lo+hi)>>1;if(arr[m].start<=g+0.01){bi=m;lo=m+1}else hi=m-1}capArr[kind]=arr;capIdx[kind]=bi}if(bi<0)return null;
 const seg=arr[bi],key=(S&&S.job)+'|'+tr+'|'+kind+'|'+bi+'|'+seg.id+'|'+seg.start+'|'+seg.end+'|'+(kind==='orig'?seg.src_text:seg.tgt_text)+'|'+(seg.dub_start||'')+'|'+(seg.dub_end||'')+'|'+capGeo.px+'|'+capGeo.fs;
 let cs=capCache.get(key);if(!cs){const nx=arr[bi+1];cs=SUB.segCues(seg,{kind:kind,lang:tr,measure:capMeasureFn(),maxPx:capGeo.px,nextStartMs:nx?Math.round(nx.start*1000):Infinity,nextSame:!!nx&&nx.speaker===seg.speaker});if(capCache.size>400)capCache.clear();capCache.set(key,cs)}
 const ms=g*1000;return cs.find(x=>ms>=x.start_ms&&ms<x.end_ms)||null}
function updateCaption(){
 const c=$('cap'),mode=capSubMode();if(!S||mode==='off'){c.hidden=true;c.textContent='';capSig='';return}
 const tr=S.track==='original'?S.lang:S.track,dubArr=tr&&tr!=='original'?S.transcripts[tr]:null,origArr=dubArr||S.transcripts[S.lang]||Object.values(S.transcripts).find(x=>x&&x.length)||null;
 capLayout();const g=globalTime(),blocks=[];
 if(mode==='dub'||mode==='dual'){if(!dubArr){if(mode==='dub'){c.hidden=true;return}}else{const q=capCues(dubArr,'dub',tr,g);if(q)blocks.push(q)}}
 if(mode==='original'||mode==='dual'){const q=capCues(origArr,'orig','und',g);if(q)blocks.unshift(q)}
 if(!blocks.length){c.hidden=true;c.textContent='';capSig='';return}
 const sig=mode+'|'+blocks.map(b=>b.id+b.revision+b.lines.join('/')).join('|')+'|'+capGeo.px+'|'+capGeo.fs;
 c.hidden=false;if(sig===capSig)return;capSig=sig;c.textContent='';c.dataset.mode=mode;
 blocks.forEach(b=>{const bl=document.createElement('span');bl.className='cb';bl.dataset.k=b.kind;bl.dataset.q=b.timing_quality;if(b.fast)bl.dataset.fast='1';
  const txt=b.lines.join(' '),rtl=SUB.isRtl(txt,b.lang);
  if(mode==='dual'){const tg=document.createElement('i');tg.className='tg';tg.textContent=b.kind==='orig'?(FX.src||'Original'):LN(b.lang);tg.setAttribute('aria-hidden','true');bl.appendChild(tg)}
  b.lines.forEach(l=>{const s=document.createElement('span');s.className='cl';s.textContent=l;s.dir=rtl?'rtl':'ltr';s.lang=b.kind==='orig'?'':b.lang;bl.appendChild(s)});
  c.appendChild(bl)});
 /* pixel-width safety net: shrink the type until no line overflows its box */
 let fs=capGeo.fs*(mode==='dual'?.85:1);c.style.setProperty('--capfs',fs+'px');
 for(let i=0;i<8&&[...c.querySelectorAll('.cb')].some(bk=>bk.scrollWidth>bk.clientWidth+1);i++){fs=Math.max(11,fs-1.5);c.style.setProperty('--capfs',fs+'px')}
 }


/* ---- /M-4 add a language mid-watch (flag DUB_FLAGS.addLang; off by default; needs POST /api/jobs/{id}/languages) ---- */
function addLangOn(){return!!(window.__DUB_FLAGS&&window.__DUB_FLAGS.addLang)||/[?&#]addlang=1\b/.test(location.href)}
function alStateText(a){if(!a)return'';const m={requesting:'Requesting…',absent:'Queued',queued:'Queued',preparing:'Preparing',partial:'Partly ready',ready:'Ready',error:a.err||'Could not start',unavailable:'Not available yet'};let t=m[a.audio_state]||'Preparing';const iv=a.ready_intervals||[];if(a.audio_state==='partial'&&iv.length)t+=' · to '+fmt(Math.max(...iv.map(x=>x[1]))/1000);if(a.lipsync_state==='ready')t+=' · Lips matched';else if(a.audio_state==='ready'||a.audio_state==='partial')t+=' · Voice only';return t}
async function addLang(code,gen){
 if(!S||!addLangOn()||!okCode(code))return;const owner=S;S.addT=S.addT||{};if(gen==null)gen=(S.pbGen=(S.pbGen||0)+1);
 const from=Math.max(0,Math.round(globalTime()*1000));S.addT[code]={audio_state:'requesting',lipsync_state:'none',ready_intervals:[],gen:gen};FX.panelRefresh();
 const stale=()=>S!==owner||gen<(S.pbGen||0);
 try{const r=await fetch(`/api/jobs/${E(S.job)}/languages`,{method:'POST',headers:hdr({'Content-Type':'application/json'}),body:JSON.stringify({lang:code,from_ms:from,playback_generation:gen})});
  let d={};try{d=await r.json()}catch(e){}
  if(stale()){if(S===owner&&S.addT[code]&&S.addT[code].gen===gen){delete S.addT[code];FX.panelRefresh()}return}   /* obsolete response: ignored; state is learned from the job poll / language.ready events */
  if(!r.ok){S.addT[code]={audio_state:'error',lipsync_state:'none',ready_intervals:[],gen:gen,err:r.status===501||r.status===404?'Not available yet':r.status===429?'Busy, try again soon':'Could not start'};if(S.wait&&S.wait.code===code){cancelWait();showToast('That language is not available yet.')}}
  else if(d.status==='obsolete'){delete S.addT[code]}
  else{const t=d.track||{};S.addT[code]={audio_state:t.audio_state||'preparing',lipsync_state:t.lipsync_state||'none',ready_intervals:Array.isArray(t.ready_intervals)?t.ready_intervals:[],gen:gen}}
 }catch(e){if(!stale()){S.addT[code]={audio_state:'error',lipsync_state:'none',ready_intervals:[],gen:gen,err:'Could not start'}}else if(S===owner&&S.addT[code]&&S.addT[code].gen===gen)delete S.addT[code]}
 FX.panelRefresh()}
function startWait(code){
 if(!S||!addLangOn()||!okCode(code))return false;
 if(!S.requestedLangs.includes(code)&&!(S.addT&&S.addT[code]))return false;
 const t=globalTime(),gen=(S.pbGen=(S.pbGen||0)+1),pre=!S.requestedLangs.includes(code),w={code:code,t:t,want:!!S.want,gen:gen,pre:pre,busy:false};
 S.wait=w;S.want=false;playSeq++;stopBuffer();try{v.pause()}catch(e){}
 stage.dataset.langwait='1';showOv('Preparing '+LN(code)+'…','Playback resumes at '+fmt(t)+' as soon as it is ready.','langwait');
 if(!pre){w.busy=true;S.want=w.want;doSwitch(code,t);w.busy=false}
 updatePlayerUI();return true}
function cancelWait(noResume){
 if(!S||!S.wait)return;const w=S.wait;S.wait=null;S.pbGen=(S.pbGen||0)+1;delete stage.dataset.langwait;if(ov.dataset.kind==='langwait')hideOv();
 if(w.pre&&w.want&&!noResume)setWant(true);updatePlayerUI()}
function tickWait(){
 if(!S||!S.wait)return;const w=S.wait;
 if(w.gen!==S.pbGen){S.wait=null;delete stage.dataset.langwait;if(ov.dataset.kind==='langwait')hideOv();return}
 if(w.pre){if(S.requestedLangs.includes(w.code)){w.pre=false;w.busy=true;S.want=w.want;doSwitch(w.code,w.t);w.busy=false}return}
 if(S.track!==w.code){S.wait=null;delete stage.dataset.langwait;return}
 if(S.chunk!=null&&S.pending==null&&!S.loading){S.wait=null;delete stage.dataset.langwait;if(ov.dataset.kind==='langwait')hideOv();offerReplay(w.code,w.t)}}
function syncAddedLangs(){
 if(!S||!S.data||!addLangOn()||!S.addT)return;let ch=false;
 for(const k of Object.keys(S.data.langs||{}))if(okCode(k)&&!S.requestedLangs.includes(k)&&(S.addT[k]||(S.wait&&S.wait.code===k))){S.requestedLangs.push(k);ch=true}
 if(ch)fillTlang()}
document.addEventListener('dubevent',e=>{const d=e&&e.detail;if(!S||!d||d.type!=='language.ready'||!okCode(d.lang))return;
 const a=S.addT&&S.addT[d.lang];if(!a)return;if(d.generation!=null&&d.generation<a.gen)return;
 S.addT[d.lang]=Object.assign({},a,{audio_state:String(d.audio_state||a.audio_state),lipsync_state:String(d.lipsync_state||a.lipsync_state),ready_intervals:Array.isArray(d.ready_intervals)?d.ready_intervals:a.ready_intervals});
 FX.panelRefresh();if(S.wait)pollTick&&pollTick()});
function replaySentence(){
 if(!S)return false;const tr=S.track==='original'?S.lang:S.track,arr=S.transcripts[tr]||S.transcripts[S.lang]||Object.values(S.transcripts).find(x=>x&&x.length);if(!arr)return false;
 const g=globalTime();let b=null;for(const x of arr){if(x.start<=g+0.01)b=x;else break}if(!b)b=arr[0];
 goTo(b.start);if(S.want&&S.chunk!=null&&!S.loading)playV();return true}
let replayT=0;
function offerReplay(code,t){
 const box=$('lchoice');box.textContent='';const b1=document.createElement('b');b1.textContent='Listening in '+LN(code)+' from '+fmt(t)+'.';box.appendChild(b1);
 const row=document.createElement('div');row.className='row';box.appendChild(row);const b=document.createElement('button');b.textContent='Replay sentence';b.dataset.ch='replay';row.appendChild(b);
 box.hidden=false;box.onclick=e=>{if(!e.target.closest('[data-ch=replay]'))return;box.hidden=true;replaySentence()};clearTimeout(replayT);replayT=setTimeout(()=>{if(box.querySelector('[data-ch=replay]'))box.hidden=true},9000)}
/* reflow cues when the stage resizes, controls show/hide or the transcript sheet opens */
(function(){const re=()=>{capSig='';if(S)updateCaption()};try{new ResizeObserver(re).observe(stage)}catch(e){addEventListener('resize',re)}
 try{new MutationObserver(()=>{if(S&&S.captions)updateCaption()}).observe(stage,{attributes:true,attributeFilter:['class']})}catch(e){}
 document.addEventListener('subreflow',e=>FX.subReflow(e.detail||{}))})();

FX.addLang=addLang;FX.startWait=startWait;FX.alText=alStateText;FX.alOpen=o=>{if(S)S.addOpen=!!o};FX.subModeName=()=>(S&&S.subMode)||'dub';
FX.alData=()=>S?{on:addLangOn(),addT:S.addT||{},open:!!S.addOpen,wait:S.wait?S.wait.code:'',name:LN,cands:(S.languages||[]).filter(l=>l&&okCode(l.code)&&!S.requestedLangs.includes(l.code)&&!(S.addT&&S.addT[l.code]))}:{on:false,addT:{},open:false,wait:'',name:x=>x,cands:[]};

let loadCtl=null;
function dropLoad(){if(loadCtl){loadCtl.abort();loadCtl=null}if(S)S.loading=false}
function attach(src,idx,t){
 dropLoad();const gen=++mediaGen,owner=S,ac=new AbortController(),sg=ac.signal,key=S.track+':'+idx+':'+ver(S.track,idx);loadCtl=ac;
 const live=()=>!sg.aborted&&S===owner&&gen===mediaGen;
 S.loading=true;S.loadIdx=idx;S.desired=t;S.curSrc=src;S.curKey=key;S.chunk=null;playSeq++;
 v.addEventListener('loadedmetadata',()=>{if(!live())return;S.loading=false;S.chunk=idx;try{v.currentTime=Math.max(0,S.desired-chunkStart(idx))}catch(e){}if(S.want&&!S.buffering){const b=bufPlan();if(b&&b.wait!=null&&b.wait>0.5)startBuffer();else playV()}updatePre();updatePlayerUI();renderSegbar()},{once:true,signal:sg});
 v.addEventListener('ended',()=>{if(!live()||S.chunk!==idx)return;const nxt=idx+1;if(nxt>=chunks().length){S.want=false;S.ended=true;stopBuffer();updatePlayerUI();return}goTo(chunkStart(nxt))},{signal:sg});
 v.addEventListener('error',()=>{if(!live())return;const fa=S.loading?S.desired:chunkStart(idx)+(v.currentTime||0);dropLoad();S.loading=false;S.chunk=null;S.pending=Math.max(chunkStart(idx),fa);S.want=false;playSeq++;stopBuffer();S.bad[key]=1;showOv('This part could not be loaded.','The audio or video file did not load. Your other parts are unaffected.','err');renderSegbar();renderParts();renderTracks();updatePlayerUI()},{signal:sg});
 FX.freeze(gen);v.addEventListener('loadeddata',()=>{if(!live())return;const rel=()=>{if(!live())return;if(v.requestVideoFrameCallback)v.requestVideoFrameCallback(()=>{if(live())FX.unfreeze(gen)});else FX.unfreeze(gen)};if(v.seeking)v.addEventListener('seeked',rel,{once:true,signal:sg});else rel()},{once:true,signal:sg});v.src=src;
}
function goTo(t){
 t=Math.max(0,Math.min(t,dur()||t));
 if(S.wait&&!S.wait.busy&&Math.abs(t-S.wait.t)>.75)cancelWait(true);
 const idx=chunkIdxOf(t),rs=ready(S.track,idx);S.ended=false;S.retrySeq=(S.retrySeq||0)+1;
 if(!rs){dropLoad();playSeq++;try{v.pause()}catch(e){}S.pending=t;S.chunk=null;S.curSrc=null;v.removeAttribute('src');try{v.load()}catch(e){}showOv(bufText(idx),'You are at Part '+(idx+1)+'. It will play when ready.','wait');postFocus(idx);highlight();updatePlayerUI();return}
 S.pending=null;if(ov.dataset.kind!=='buf')hideOv();const src=srcFor(S.track,idx);
 const same=(S.loading?S.loadIdx===idx:S.chunk===idx)&&v.getAttribute('src')===src;
 if(!same)attach(src,idx,t);
 else if(S.loading)S.desired=t;
 else{try{v.currentTime=Math.max(0,t-chunkStart(idx))}catch(e){}}
 postFocus(idx);highlight();updatePlayerUI();
}
function postFocus(idx,force){
 const key=S.job+':'+S.track+':'+idx;if(S.track!=='original'&&(key!==lastPost||force)){lastPost=key;lastPostAt=Date.now();fetch(`/api/jobs/${E(S.job)}/focus`,{method:'POST',headers:hdr({'Content-Type':'application/json'}),body:JSON.stringify({lang:S.track,chunk:idx})}).catch(()=>{})}
}
function updatePre(){
 if(S.chunk==null)return;const nxt=S.chunk+1;
 if(S.track!=='original'&&ready(S.track,nxt))pre.src=srcFor(S.track,nxt);else pre.removeAttribute('src');
}
function doSwitch(code,t){
 S.track=code;if(code!=='original'&&!S.transcripts[code])loadTranscript(code);
 goTo(t);v.playbackRate=S.rate;renderTracks();renderSegbar();renderParts();updateCaption();
 FX.switched(code==='original'?'Original':LN(code),code==='original'?'Original audio':partState(code,chunkIdxOf(t)).t);
}
function setTrack(code){
 if(!S||!S.data||code===S.track)return;const t=globalTime(),idx=chunkIdxOf(t);
 if(S.wait&&S.wait.code!==code)cancelWait();
 if(!ready(code,idx)){if(code!=='original'&&startWait(code))return;offerChoice(code,t,idx);return}
 if(S.wait&&S.wait.code===code){S.wait=null;delete stage.dataset.langwait;if(ov.dataset.kind==='langwait')hideOv()}
 $('lchoice').hidden=true;doSwitch(code,t);
}
function offerChoice(code,t,idx){
 const cs=chunks();let last=-1;for(const c of cs){if(c.idx<idx&&ready(code,c.idx))last=c.idx}
 const box=$('lchoice');
 const nm=code==='original'?'Original':LN(code);
 box.textContent='';
 const b1=document.createElement('b');box.appendChild(b1);
 if(last>=0){b1.textContent=nm+' is not ready at '+fmt(t)+' yet.';const c=cs.find(x=>x.idx===last),sp=document.createElement('span');sp.className='mut';sp.textContent='Ready through '+fmt(c.start+c.dur)+'.';box.appendChild(sp)}else b1.textContent='The first '+nm+' part is still being created.';
 const row=document.createElement('div');row.className='row';box.appendChild(row);
 const mk=(label,ch)=>{const b=document.createElement('button');b.textContent=label;b.dataset.ch=ch;row.appendChild(b)};
 mk('Wait here','wait');if(last>=0)mk('Jump to latest ready part','jump');if(code!=='original'&&ready('original',idx))mk('Play original','orig');mk('Cancel','cancel');
 box.hidden=false;
 box.onclick=e=>{const b=e.target.closest('[data-ch]');if(!b)return;box.hidden=true;
  if(b.dataset.ch==='wait')doSwitch(code,t);else if(b.dataset.ch==='jump'){const c=cs.find(x=>x.idx===last);doSwitch(code,c.start)}else if(b.dataset.ch==='orig')doSwitch('original',t)};
 const f=box.querySelector('button');if(f)f.focus();
}
function setWant(w){
 if(!S){return}
 if(S.wait&&S.wait.pre){S.wait.want=!!w;updatePlayerUI();return}
 if(w&&S.data&&!chunks().length){showToast('There are no playable parts yet.');return}
 if(w&&S.ended){S.ended=false;S.want=true;goTo(0);updatePlayerUI();return}
 if(w&&S.chunk!=null&&S.pending==null&&!S.buffering){const b=bufPlan();if(b&&b.wait!=null&&b.wait>0.5){startBuffer();return}}
 if(w&&S.pending!=null&&S.bad){const bi=chunkIdxOf(S.pending),bk=S.track+':'+bi+':'+ver(S.track,bi);if(S.bad[bk]){delete S.bad[bk];if(ov.dataset.kind==='err')hideOv()}}
 S.want=w;if(w){if(S.pending!=null)goTo(S.pending);else if(S.chunk!=null){playV();updatePre()}else if(S.data&&chunks().length)goTo(globalTime()||0)}else{playSeq++;stopBuffer();v.pause();if(ov.dataset.kind!=='err')hideOv()}
 updatePlayerUI();
}
function togglePlay(){setWant(!S.want);FX.feedback(S.want?'play':'pause')}

let pollOn=false,pollSeq=0,pollFails=0,pollBusy=null,pollGen=0;
function setConn(bad){if(!S)return;S.conn=bad;renderBanner()}
function startPoll(){stopPoll();pollOn=true;const gen=++pollGen;const loop=async()=>{if(!pollOn||gen!==pollGen)return;try{await pollTick()}catch(e){}if(pollOn&&gen===pollGen)pollT=setTimeout(loop,3000)};pollT=setTimeout(loop,3000)}
function stopPoll(){pollOn=false;pollGen++;abortPoll();if(pollT){clearTimeout(pollT);pollT=null}}
function abortPoll(){if(pollBusy){try{pollBusy.ac.abort()}catch(e){}pollBusy=null}}
function pollTick(){
 if(!S)return Promise.resolve();if(pollBusy&&pollBusy.owner===S)return pollBusy.p;
 const tok={owner:S,ac:new AbortController()};pollBusy=tok;tok.p=pollRun(tok).finally(()=>{if(pollBusy===tok)pollBusy=null});return tok.p;
}
async function pollRun(tok){
 const my=++pollSeq,owner=tok.owner,job=owner.job;
 try{
  const ac=tok.ac,tm=setTimeout(()=>ac.abort(),15000);let d;try{d=await fetch(`/api/jobs/${E(job)}`,{signal:ac.signal}).then(r=>{if(!r.ok)throw Error(r.status);return r.json()})}finally{clearTimeout(tm)}
  if(my!==pollSeq||S!==owner)return;
  pollFails=0;setConn(false);S.data=normJob(d);S.dataAt=performance.now();
  if(!S.requestedLangs.length){const ks=Object.keys(S.data.langs);if(ks.length){S.requestedLangs=ks;S.lang=ks[0];fillTlang();loadTranscript()}}
  if(S.track!=='original'&&S.chunk!=null&&Date.now()-lastPostAt>60000)postFocus(S.chunk,true);
  if(S.data.state==='error'){stopPoll();stopBuffer();S.pending=null;if(S.chunk==null){S.want=false;if(ov.dataset.kind==='wait')hideOv()}showToast('Job error — some parts failed',6000,true)}
  renderProg();
  if(S.pending!=null&&ready(S.track,chunkIdxOf(S.pending))){const pt=S.pending;S.pending=null;goTo(pt)}
  if(S.chunk!=null&&!S.loading&&S.track!=='original'&&ready(S.track,S.chunk)){const wsrc=srcFor(S.track,S.chunk);if(v.getAttribute('src')!==wsrc)attach(wsrc,S.chunk,globalTime())}
  updatePre();
  if(S.chunk!=null&&!ready(S.track,S.chunk)&&!(S.bad&&S.bad[S.curKey])){S.pending=globalTime();dropLoad();mediaGen++;playSeq++;try{v.pause()}catch(e){}S.chunk=null;S.curSrc=null;v.removeAttribute('src');try{v.load()}catch(e){}showOv('Dubbing the next part…','Playback will continue when this part is ready.','wait')}
  if(S.chunk==null&&S.pending!=null&&ov.style.display==='grid'&&ov.dataset.kind==='wait')ovt.textContent=bufText(chunkIdxOf(S.pending));
 }catch(e){if(S===owner&&my===pollSeq&&++pollFails>=2)setConn(true)}
}

// ---- transcript ----
function rowState(seg){
 const c=chunkInfo(S.lang,+seg.chunk);if(!c)return '';
 if(c.editing)return '<span class="pill info">Updating</span>';
 if(c.state==='error')return '<span class="pill bad">Part failed</span>';
 if(c.state==='done'&&c.has_media)return (c.sync&&c.sync.lipsync===true)?'<span class="pill ok">Lip-matched</span>':(S&&S.data&&S.data.lipsync===false)?'<span class="pill info">Aligned audio</span>':'<span class="pill warn">Voice ready</span>';
 return '<span class=pill>Not ready</span>';
}
function renderTranscriptRows(){
 if(!S||S.lang==='original'||!S.lang||!S.data||!S.data.langs[S.lang]){segsBox.innerHTML='<div class=empty>Choose a dubbed language to view or edit its translation.</div>';return}
 if(S.tErr[S.lang]){segsBox.innerHTML='<div class=empty>Transcript could not be loaded.<div style="margin-top:10px"><button id=tretry>Retry</button> <button id=tcopy>Copy details</button></div></div>';$('tretry').onclick=()=>loadTranscript();$('tcopy').onclick=()=>copyText(`job ${S.job} ${S.lang}: ${S.tErr[S.lang]}`);return}
 const d=S.transcripts[S.lang];if(!d){segsBox.innerHTML='<div aria-hidden=true>'+'<div class=sk></div>'.repeat(5)+'</div><div class=vh>Loading transcript…</div>';return}
 if(!d.length){segsBox.innerHTML='<div class=empty>Transcript lines will appear when speech has been found.</div>';return}
 segsBox.innerHTML=d.map(seg=>{
  const sid=(seg.speaker|0)+1,lang=esc(S.lang);
  return `<div class="seg-row" role=group aria-label="Line at ${fmt(seg.start)}" data-lang="${lang}" data-id="${esc(seg.id)}" data-chunk="${esc(seg.chunk)}" data-start="${seg.start.toFixed(2)}" data-end="${seg.end.toFixed(2)}"><div class=rh><button class=tm title="Seek to ${fmt(seg.start)}">${fmt(seg.start)}</button><span class=spk><i class=sw style="background:${col(seg.speaker)}" aria-hidden=true>S${sid}</i>Speaker ${sid} · ${GT(seg.gender)}</span>${rowState(seg)}</div><div class=sbox><div class=src dir=auto></div><div class=tgt dir=auto lang="${lang}"><span class=tx></span>${seg.edited?' <span class=badge>Edited</span>':''}${seg.fallback?` <span class="badge fb" title="We kept the original audio for this line because the dubbed voice could not be used. You can edit the translation and save to try again.">original audio — <span class=fbr></span></span>`:''}</div>${seg.fallback?'<div class=fbx>We kept the original audio for this line because the dubbed voice could not be used. You can edit the translation and save to try again.</div>':''}<div class=sact>${seg.history_len>0?'<button data-act=undo>Undo</button>':''}<button data-act=edit>Edit</button></div>${xpBar(seg)}</div></div>`;

 }).join('');
 const rows=segsBox.querySelectorAll('.seg-row');
 d.forEach((seg,i)=>{const r=rows[i];if(!r)return;r.querySelector('.src').textContent=FX.sc(seg.src_text==null?'':String(seg.src_text));r.querySelector('.tx').textContent=seg.tgt_text==null?'':String(seg.tgt_text);const f=r.querySelector('.fbr');if(f)f.textContent=seg.fallback_reason||'reason unavailable'});
 FX.rows();xpRestore();ctlProbe();
}
function showStale(on){let n=$('tstale');if(!n){n=document.createElement('div');n.id='tstale';n.className='sub';n.style.cssText='padding:6px 16px;font-size:12px;border-bottom:1px solid var(--bd)';n.innerHTML='Showing the last loaded transcript. <button type=button id=tstaler style="min-height:32px;padding:0 10px">Retry</button>';segsBox.parentNode.insertBefore(n,segsBox);$('tstaler').onclick=()=>loadTranscript()}n.hidden=!on}
async function loadTranscript(LG,opts){
 if(!S)return;opts=opts||{};const owner=S,J=S.job;LG=(typeof LG==='string'&&LG)?LG:S.lang;
 if(!LG||LG==='original'){if(!LG||LG===S.lang)renderTranscriptRows();return}
 const seq=(S.tseq[LG]=(S.tseq[LG]||0)+1);
 S.tErr[LG]=null;if(LG===S.lang&&!S.transcripts[LG])renderTranscriptRows()
 try{const d=await url(`/api/jobs/${E(J)}/${E(LG)}/segments`);if(S!==owner||S.tseq[LG]!==seq)return;
  S.transcripts[LG]=normSegs(d);capCache.clear();capSig='';S.tStale[LG]=false;if(S.lang===LG)showStale(false);
  if(S.lang===LG){if(opts.force||!segsBox.querySelector('.ed'))renderTranscriptRows();highlight()}else updateCaption();
 }catch(e){if(S===owner&&S.tseq[LG]===seq){if(S.transcripts[LG])S.tStale[LG]=true;else S.tErr[LG]=String(e.message||e);if(S.lang===LG){if(!segsBox.querySelector('.ed')||!S.tStale[LG])renderTranscriptRows();showStale(!!S.tStale[LG])}}}
}
const rowOf=(id,chunk)=>[...segsBox.querySelectorAll('.seg-row')].find(r=>r.dataset.id===String(id)&&r.dataset.chunk===String(chunk));
const segOf=(id,chunk)=>(S.transcripts[S.lang]||[]).find(s=>String(s.id)===String(id)&&String(s.chunk)===String(chunk));
async function reloadPlayback(chunk,J,LG){
 const cur=()=>S&&S.job===J&&S.track===LG&&(S.loading?S.loadIdx:S.chunk)===+chunk;
 if(!cur())return;if(pollBusy&&pollBusy.owner===S){try{await pollBusy.p}catch(e){}}if(!cur())return;
 await pollTick();if(!cur())return;
 const gt=globalTime();S.chunk=null;S.curSrc=null;v.removeAttribute('src');goTo(gt)
}
async function withLock(fn){
 if(!S)return;if(S.mut){showToast('Please wait for the current change to finish.');return}
 const o=S;o.mut=true;segsBox.setAttribute('aria-busy','true');
 try{await fn()}finally{o.mut=false;if(!S||S===o||!S.mut)segsBox.removeAttribute('aria-busy')}
}
const editSave=(id,chunk,text)=>withLock(async()=>{
 const J=S.job,LG=S.lang,own=S;showToast('Saving and rebuilding voice…',8000);
 let r=null,em='',conf=false;
 try{const x=await fetch(`/api/jobs/${E(J)}/${E(LG)}/chunks/${E(chunk)}/segments/${E(id)}`,{method:'POST',headers:hdr({'Content-Type':'application/json'}),body:JSON.stringify({text})});if(x.ok)r=await x.json();else{em=await errMsg(x,'edit failed');if(x.status===409)conf=true}}catch(e){em='network error: '+e}
 if(S!==own)return;
 if(conf){{const rw=rowOf(id,chunk);if(rw){rw.removeAttribute('aria-busy');rw.querySelectorAll('textarea,[data-act]').forEach(b=>b.disabled=false);const sv=rw.querySelector('[data-act=save]');if(sv)sv.textContent='Save'}}showToast('This line was changed elsewhere, so your edit was not saved. The latest version is loaded and your text is kept below.',9000,true);await loadTranscript(LG,{force:true});setTimeout(()=>{if(S===own&&S.lang===LG&&!S.mut){if(!segsBox.querySelector('.seg-row.editing'))startEdit(id,chunk);const ta=segsBox.querySelector('.seg-row.editing .ed textarea');if(ta){ta.disabled=false;ta.value=text;upCount(ta)}}},30);return}
 if(!r||r.ok===false){showToast('The translation could not be saved. Nothing was changed.'+(em?' '+String(em).replace(/\s+/g,' ').slice(0,200):''),8000,true);await loadTranscript(LG,{force:true});return}
 {pollSeq++;const ck=((S.data&&S.data.langs[LG])||{}).chunks,cc=ck&&ck.find(x=>x.idx===+chunk);if(cc&&r.version!=null)cc.version=num(r.version)|0}
 await loadTranscript(LG,{force:true});if(S!==own)return;showToast('Translation saved. This part has been updated.');
 if(S.lang===LG){const rr=rowOf(id,chunk);if(rr){const b=rr.querySelector('[data-act=edit]');if(b)b.focus({preventScroll:true})}}
 await reloadPlayback(chunk,J,LG);
});
const editUndo=(id,chunk)=>withLock(async()=>{
 const J=S.job,LG=S.lang,own=S;showToast('Reverting…',8000);
 let r=null,em='';
 try{const x=await fetch(`/api/jobs/${E(J)}/${E(LG)}/chunks/${E(chunk)}/segments/${E(id)}/undo`,{method:'POST',headers:hdr()});if(x.ok)r=await x.json();else em=await errMsg(x,'undo failed')}catch(e){em='network error: '+e}
 if(S!==own)return;
 if(!r||r.ok===false){showToast('Undo could not be completed.'+(em?' '+String(em).replace(/\s+/g,' ').slice(0,200):''),8000,true);await loadTranscript(LG,{force:true});return}
 {pollSeq++;const ck=((S.data&&S.data.langs[LG])||{}).chunks,cc=ck&&ck.find(x=>x.idx===+chunk);if(cc&&r.version!=null)cc.version=num(r.version)|0}
 await loadTranscript(LG,{force:true});if(S!==own)return;showToast('The previous translation was restored.');
 await reloadPlayback(chunk,J,LG);
});
function startEdit(id,chunk){
 if(S.mut)return;const row=rowOf(id,chunk);if(!row||row.querySelector('.ed'))return;
 const seg=segOf(id,chunk);if(!seg)return;
 const ed=document.createElement('div');ed.className='ed';
 ed.innerHTML='<label class=lab>Edit translated line</label><textarea dir=auto></textarea><div class=edinfo><span class=cnt></span><span class=w></span></div><div class=sact><button data-act=save class=btn disabled>Save</button><button data-act=cancel>Cancel</button>'+(seg.history_len>0?'<button data-act=undo>Undo</button>':'')+'</div>';
 const ta=ed.querySelector('textarea'),tid='ta_'+(++edSeq);ta.id=tid;ed.querySelector('label').htmlFor=tid;ta.lang=S.lang;
 ta.value=seg.tgt_text;ta.dataset.orig=ta.value;
 row.classList.add('editing');row.querySelector('.sbox').appendChild(ed);ta.focus();ta.setSelectionRange(ta.value.length,ta.value.length);upCount(ta);
}
function cancelEdit(row){const ed=row.querySelector('.ed');if(ed)ed.remove();row.classList.remove('editing');const b=row.querySelector('.sbox>.sact [data-act=edit]');if(b)b.focus({preventScroll:true})}
function upCount(ta){const n=Array.from(ta.value).length,ed=ta.closest('.ed');ed.querySelector('.cnt').textContent=n+' characters';ed.querySelector('.w').textContent='';ed.querySelector('[data-act=save]').disabled=ta.value.trim()===''||ta.value===ta.dataset.orig}

function applyView(){
 view('studio');showStale(false);segsBox.removeAttribute('aria-busy');$('spdbtn').innerHTML='<b>1×</b>';$('spdbtn').setAttribute('aria-label','Playback speed 1×');v.playbackRate=1;v.defaultPlaybackRate=1;$('follow').setAttribute('aria-pressed','true');$('follow').textContent='Following playback';const sel=S.requestedLangs;
 if(!S.data){[tracksBox,$('parts')].forEach(e=>{e.innerHTML='';e.__h=null});segbar.innerHTML='';$('qsum').innerHTML='';$('syncb').innerHTML=''}
 fillTlang();setTab('transcript');
 $('capbtn').setAttribute('aria-pressed',String(S.captions));
 renderBanner();renderTranscriptRows();renderTracks();renderParts();renderSegbar();renderSync();updatePlayerUI();
 FX.meta(S.data?{url:S.data.url,job:S.job,duration:S.data.duration,parts:chunks().length,state:S.data.state,langs:S.requestedLangs.map(LN)}:null);
 window.scrollTo(0,0);ctlOpen();
}
function setTab(n){$('studio').dataset.tab=n;document.querySelectorAll('#mtabs button').forEach(b=>{const on=b.dataset.tab===n;b.setAttribute('aria-selected',String(on));b.tabIndex=on?0:-1})}
function resetStudio(){
 FX.reset();ctlClose();segsBox.removeAttribute('aria-busy');stopPoll();if(S&&S.bufT)clearInterval(S.bufT);dropLoad();mediaGen++;createGen++;setCreating(false);seekPreview=null;try{v.pause();v.removeAttribute('src');v.load()}catch(e){}pre.removeAttribute('src');hideOv();
 S=null;history.replaceState(null,'',location.pathname+location.search);view('landing');$('banner').hidden=true;$('formmsg').hidden=true;$('lchoice').hidden=true;$('cap').hidden=true;stage.classList.add('idle');
}
function curRow(){return segsBox.querySelector('.seg-row.cur')}
function seekRel(dt){goTo(Math.max(0,Math.min(globalTime()+dt,dur())))}
function partJump(dir){const g=globalTime(),idx=chunkIdxOf(g);if(dir>0){const nx=chunks().find(c=>c.idx===idx+1);if(nx)goTo(nx.start)}else{const cs=chunkStart(idx);const t=(g-cs<2&&idx>0)?chunkStart(idx-1):cs;goTo(t)}}
function cycleLang(){const order=['original',...S.requestedLangs],i=order.indexOf(S.track);setTrack(order[(i+1)%order.length])}
function toggleFs(){if(document.fullscreenElement)document.exitFullscreen();else if(stage.requestFullscreen)stage.requestFullscreen().catch(()=>{})}
function setMute(m){v.muted=m;$('muteic').innerHTML=FX.ic(m?'muted':'volume');$('mutebtn').setAttribute('aria-label',m?'Unmute':'Mute')}
function toggleCaps(){if(S.track==='original'&&!S.captions&&!(S.lang&&S.lang!=='original'&&S.transcripts[S.lang])){showToast('Captions need a loaded translation. Choose a language in the transcript first.');return}S.captions=!S.captions;$('capbtn').setAttribute('aria-pressed',String(S.captions));$('capbtn').setAttribute('aria-label',S.captions?'Hide captions':'Show captions');updateCaption();updatePlayerUI()}
function setSubMode(m){if(!S||!['off','original','dub','dual'].includes(m))return false;if(m==='off'){if(S.captions)toggleCaps();return!S.captions}const was=S.captions;S.subMode=m;if(!was){toggleCaps();if(!S.captions)return false}else{capSig='';updateCaption()}return true}
const SPEEDS=[0.75,1,1.25,1.5,2];

const isInteractive=t=>!!(t&&t.closest&&t.closest('button,a[href],summary,input,textarea,select,[role=button],[role=tab],[contenteditable=""],[contenteditable=true]'));
function initEvents(){
 FX.setTrack=code=>{if(!S||!S.data||!(code==='original'||S.requestedLangs.includes(code)))return false;if(code===S.track)return true;setTrack(code);return S.track===code?true:($('lchoice').hidden?false:'deferred')};FX.setSubMode=setSubMode;FX.subMode=()=>capSubMode();FX.addLangState=()=>S?{pbGen:S.pbGen||0,addT:JSON.parse(JSON.stringify(S.addT||{})),wait:S.wait?{code:S.wait.code,t:S.wait.t,want:S.wait.want,gen:S.wait.gen,pre:!!S.wait.pre}:null,want:!!S.want,track:S.track}:null;FX.subReflow=d=>{if(S){S.sheet=d||{};capSig='';capCache.clear();updateCaption()}};FX.replaySentence=()=>replaySentence();FX.toggleCaps=()=>{if(!S)return false;const b=S.captions;toggleCaps();return S.captions!==b};
 FX.formWhy=()=>{const raw=$('url').value.trim();if(!ytOk(raw))return['url',raw?'Enter a valid YouTube link.':'Paste a YouTube link first.'];if(!picked.length)return['pickbtn','Choose at least one language.'];const rc=rangeCheck();if(!rc.ok)return[document.querySelector('input[name=rng]:checked').value==='clip'?'start':'url',rc.msg];if(!$('consent').checked)return['consent','Confirm you have the right to dub this video.'];return null};
 $('skb').addEventListener('click',()=>{if(S){seekRel(-10);FX.feedback('back')}});$('skf').addEventListener('click',()=>{if(S){seekRel(10);FX.feedback('fwd')}});$('audbtn').addEventListener('click',()=>FX.togglePanel());
 form.addEventListener('submit',createJob);
 document.querySelectorAll('.demo').forEach((b,i)=>b.addEventListener('click',()=>openDemo(i)));
 $('pickbtn').addEventListener('click',()=>{const o=$('pickpanel').hidden;$('pickpanel').hidden=!o;$('pickbtn').setAttribute('aria-expanded',String(o));if(o)$('psearch').focus()});
 $('psearch').addEventListener('input',renderPick);
 $('pickdone').addEventListener('click',()=>{$('pickpanel').hidden=true;$('pickbtn').setAttribute('aria-expanded','false');$('pickbtn').focus()});
 $('picklist').addEventListener('change',e=>{const i=e.target;if(i.type!=='checkbox')return;if(i.checked){if(picked.length<3)picked.push(i.value)}else picked=picked.filter(c=>c!==i.value);fe('lange','');renderPick();updateSubmit()});
 document.querySelectorAll('input[name=rng]').forEach(r=>r.addEventListener('change',()=>{$('clip').hidden=document.querySelector('input[name=rng]:checked').value==='whole';fe('rnge','');updateSubmit()}));
 ['url','start','end'].forEach(id=>$(id).addEventListener('input',updateSubmit));$('consent').addEventListener('change',updateSubmit);
 $('url').addEventListener('blur',()=>{const r=$('url').value.trim();if(r&&!ytOk(r)){fe('urle','Enter a valid YouTube link.');$('url').setAttribute('aria-invalid','true')}else{fe('urle','');$('url').removeAttribute('aria-invalid')}});
 $('end').addEventListener('blur',()=>{const r=rangeCheck();fe('rnge',r.ok||(!$('start').value.trim()&&!$('end').value.trim())?'':r.msg)});
 ['start','end'].forEach(id=>$(id).addEventListener('blur',()=>{const t=parseT($(id).value);if(isFinite(t)&&$(id).value.trim()&&!/\./.test($(id).value))$(id).value=fmt(t)}));
 $('home').addEventListener('click',()=>{if(S)resetStudio()});
 $('helpx').addEventListener('click',()=>$('help').close());
 $('mpp').addEventListener('click',()=>pp.click());$('mcc').addEventListener('click',()=>$('capbtn').click());$('mlang').addEventListener('click',()=>{if(S)cycleLang()});
 $('newbtn').addEventListener('click',resetStudio);
 $('helpbtn').addEventListener('click',()=>$('help').showModal());
 $('helpclose').addEventListener('click',()=>$('help').close());
 $('helpnew').addEventListener('click',()=>{$('help').close();resetStudio()});
 $('themebtn').addEventListener('click',()=>{FX.themeFade();const n=document.documentElement.dataset.theme==='dark'?'light':'dark';document.documentElement.dataset.theme=n;try{localStorage.setItem('dub-theme',n)}catch(e){}themeLbl()});
 // player
 bar.addEventListener('input',()=>{if(!S)return;seekPreview=num(bar.value);tlabel.textContent=fmt(seekPreview)+' / '+fmt(dur());$('fill').style.width=(dur()?seekPreview/dur()*100:0)+'%'});
 bar.addEventListener('change',()=>{if(!S)return;const t=seekPreview!=null?seekPreview:num(bar.value);seekPreview=null;goTo(t)});
 const prev=$('prev');
 const showPrev=x=>{if(!S||!dur())return;const r=bar.getBoundingClientRect(),f=Math.max(0,Math.min(1,(x-r.left)/r.width)),t=f*dur(),idx=chunkIdxOf(t);
  const ps=S.track==='original'?(ready('original',idx)?'Original':'Not ready'):partState(S.track,idx).t;prev.textContent=fmt(t)+' · Part '+(idx+1)+' · '+ps;prev.style.left=Math.max(60,Math.min(r.width-60,f*r.width))+'px';prev.hidden=false};
 bar.addEventListener('pointermove',e=>showPrev(e.clientX));bar.addEventListener('pointerleave',()=>{prev.hidden=true});bar.addEventListener('pointerup',()=>setTimeout(()=>{prev.hidden=true;bar.blur()},400));
 bar.addEventListener('keydown',e=>{if(e.key==='ArrowLeft'||e.key==='ArrowRight')e.stopPropagation()});
 v.addEventListener('timeupdate',()=>{if(!S)return;updatePlayerUI();highlight()});
 pp.addEventListener('click',togglePlay);$('bigplay').addEventListener('click',()=>{if(S)setWant(true)});
 stage.addEventListener('click',e=>{if(e.target===v&&S)togglePlay()});
 tracksBox.addEventListener('click',e=>{const b=e.target.closest('.tt');if(b)setTrack(b.dataset.trk)});
 $('parts').addEventListener('click',e=>{const b=e.target.closest('.pc');if(!b||!S)return;const l=b.dataset.lang,i=+b.dataset.part;
  if(l!==S.track){if(!ready(l,i)){offerChoice(l,chunkStart(i),i);return}doSwitch(l,chunkStart(i));return}
  goTo(chunkStart(i));renderTracks();renderSegbar()});
 $('mutebtn').addEventListener('click',()=>setMute(!v.muted));
 $('vol').addEventListener('input',()=>{v.volume=+$('vol').value;$('vol').setAttribute('aria-label','Volume '+Math.round(v.volume*100)+'%');if(v.volume>0&&v.muted)setMute(false)});
 $('spdbtn').addEventListener('click',()=>{if(!S)return;const i=(SPEEDS.indexOf(S.rate)+1)%SPEEDS.length;S.rate=SPEEDS[i];v.defaultPlaybackRate=S.rate;v.playbackRate=S.rate;$('spdbtn').innerHTML='<b>'+S.rate+'×</b>';$('spdbtn').setAttribute('aria-label','Playback speed '+S.rate+'×')});
 v.addEventListener('loadedmetadata',()=>{if(S){v.playbackRate=S.rate}});
 $('capbtn').addEventListener('click',()=>S&&toggleCaps());
 $('fsbtn').addEventListener('click',toggleFs);
 document.addEventListener('fullscreenchange',()=>{$('fsbtn').setAttribute('aria-label',document.fullscreenElement?'Exit fullscreen':'Enter fullscreen')});
 if(!stage.requestFullscreen)$('fsbtn').hidden=true;
 document.querySelectorAll('#mtabs button').forEach(b=>b.addEventListener('click',()=>setTab(b.dataset.tab)));
 $('mtabs').addEventListener('keydown',e=>{if(!['ArrowLeft','ArrowRight','Home','End'].includes(e.key))return;const bs=[...document.querySelectorAll('#mtabs button')].filter(x=>x.offsetParent),i=bs.indexOf(document.activeElement);if(i<0)return;e.preventDefault();const n=e.key==='Home'?0:e.key==='End'?bs.length-1:(i+(e.key==='ArrowRight'?1:-1)+bs.length)%bs.length;bs[n].focus();setTab(bs[n].dataset.tab)});
 // transcript
 tlang.addEventListener('change',()=>{const ta=segsBox.querySelector('.ed textarea');
  if(ta&&ta.value!==ta.dataset.orig&&!confirm('You have an unsaved edit. Discard it and switch language?')){tlang.value=S.lang;return}
  S.lang=tlang.value;renderTranscriptRows();showStale(!!S.tStale[S.lang]);loadTranscript();renderTracks()});
 $('follow').addEventListener('click',()=>{S.follow=!S.follow;$('follow').setAttribute('aria-pressed',String(S.follow));$('follow').textContent=S.follow?'Following playback':'Follow playback';if(S.follow){const r=curRow();if(r)followScroll(r)}});
 const pauseFollow=()=>{if(S&&S.follow){S.follow=false;$('follow').setAttribute('aria-pressed','false');$('follow').textContent='Follow playback'}};
 segsBox.addEventListener('pointerdown',e=>{if(e.target===segsBox)pauseFollow()});segsBox.addEventListener('wheel',pauseFollow,{passive:true});segsBox.addEventListener('touchmove',pauseFollow,{passive:true});
 segsBox.addEventListener('keydown',e=>{if(['PageUp','PageDown','Home','End'].includes(e.key)&&!/TEXTAREA/.test(e.target.tagName))pauseFollow()});
 segsBox.addEventListener('input',e=>{if(e.target.matches('textarea')){e.stopPropagation();upCount(e.target)}});
 segsBox.addEventListener('keydown',e=>{if(e.key==='Escape'&&e.target.closest('.ed')){e.stopPropagation();e.preventDefault();cancelEdit(e.target.closest('.seg-row'))}});
 segsBox.addEventListener('click',async e=>{
  const tm=e.target.closest('.tm'),act=e.target.closest('[data-act]');
  if(!act){if(tm){const row=tm.closest('.seg-row');if(row)goTo(parseFloat(row.dataset.start)||0);return}return}
  const row=e.target.closest('.seg-row');const id=row&&row.dataset.id,chunk=row&&row.dataset.chunk;
  if(!row)return;
  if(row.dataset.lang!==S.lang){showToast('The transcript language changed. Try again.');return}
  if(act.dataset.act==='edit')startEdit(id,chunk);
  else if(act.dataset.act==='cancel')cancelEdit(row);
  else if(act.dataset.act==='save'){if(S.mut){showToast('Please wait for the current change to finish.');return}const ta=row.querySelector('.ed textarea');if(!ta)return;const txt=ta.value;row.setAttribute('aria-busy','true');ta.disabled=true;row.querySelectorAll('[data-act]').forEach(b=>b.disabled=true);act.textContent='Saving and rebuilding voice…';await editSave(id,chunk,txt)}
  else if(act.dataset.act==='undo'){if(S.mut){showToast('Please wait for the current change to finish.');return}row.querySelectorAll('[data-act]').forEach(b=>b.disabled=true);await editUndo(id,chunk)}
 });
 // keyboard
 document.addEventListener('keydown',e=>{
  if(e.defaultPrevented||!S||document.body.dataset.view!=='studio')return;
  if(e.key==='Escape'){if(document.fullscreenElement)document.exitFullscreen();return}
  const t=e.target;if(t&&(/^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName)||t.isContentEditable))return;
  if(t&&t.closest&&t.closest('#audpanel,#lchoice,#banner,[role=tablist],#hmore,#chat'))return;
  if(e.ctrlKey||e.metaKey||e.altKey||document.querySelector('dialog[open]'))return;
  const k=e.key;
  if(e.code==='Space'||k===' '){if(isInteractive(t))return;e.preventDefault();togglePlay();return}
  if(t&&t.tagName==='BUTTON'&&(k==='Enter'))return;
  if(k==='ArrowLeft'||k==='ArrowRight'){if(t===bar)return;e.preventDefault();if(e.shiftKey)partJump(k==='ArrowRight'?1:-1);else seekRel(k==='ArrowRight'?5:-5);return}
  const kk=k.toLowerCase();
  if(k==='?'||(k==='/'&&e.shiftKey)){e.preventDefault();$('help').showModal();return}
  if(e.shiftKey)return;
  if(kk==='l'){cycleLang();FX.openPanel(true)}else if(kk==='o'){setTrack('original')}
  else if(kk==='f'){toggleFs()}else if(kk==='m'){setMute(!v.muted)}else if(kk==='c'){toggleCaps()}
  else if(kk==='e'){if(S.lang==='original'){showToast('Choose a dubbed language to edit.');return}const r=curRow();if(r){startEdit(r.dataset.id,r.dataset.chunk);e.preventDefault()}else showToast('Play or select a transcript line first, then press E to edit it.')}
  else if(kk==='u'){if(S.mut){showToast('Please wait for the current change to finish.');return}const r=curRow();if(r&&r.querySelector('.sbox>.sact [data-act=undo]'))editUndo(r.dataset.id,r.dataset.chunk)}
 });
}
function themeLbl(){const d=document.documentElement.dataset.theme==='dark';$('themebtn').setAttribute('aria-label',d?'Use light theme':'Use dark theme');$('themeic').innerHTML=d?'<circle cx=12 cy=12 r=4.5 /><path d="M12 2v2M12 20v2M2 12h2M20 12h2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>':'<path d="M20 14.5A8 8 0 1 1 9.5 4a6.5 6.5 0 0 0 10.5 10.5z"/>'}
async function init(){
 themeLbl();initEvents();renderPick();updateSubmit();
 const m=location.hash.match(/job=(\w+)/);
 loadLangs().then(()=>{});
 if(m){
  const job=m[1];let d=null,nf=false;try{d=normJob(await url(`/api/jobs/${E(job)}`))}catch(e){nf=String(e.message)==='404'}
  if(d&&Object.keys(d.langs).length){const ks=Object.keys(d.langs);newState({job_id:job,langs:ks},ks);S.data=d;S.dataAt=performance.now();applyView();startPoll();pollTick();loadTranscript()}
  else if(d&&(d.state==='planning'||d.state==='queued')){newState({job_id:job,langs:[]},[]);S.lang=null;S.data=d;S.dataAt=performance.now();applyView();startPoll();pollTick()}
  else{$('formmsg').hidden=false;$('formmsg').textContent=nf?'Job not found.':'Could not reach the studio. Check your connection and reload.'}
 }
}
/* ---- M-5: control plane client (flagged: only active when GET /api/ctl/config says enabled) ---- */
const CTL={on:false,expr:false,chat:false,job:null,seq:0,gen:0,ac:null,cfgP:null},PV={},OPS={},ISL=window.__ISL=[];
const XI=[['angrier','Angrier'],['sadder','Sadder'],['calmer','Calmer'],['excited','Excited']],XL=Object.fromEntries(XI);
const STAGE_TXT={queued:'Queued…',synthesizing:'Synthesizing voice…',audio_ready:'Audio ready',lip_syncing:'Lip-syncing…',complete:'Complete',error:'That did not work'};
const mediaUrl=u=>{try{if(typeof u!=='string'||!u||u.length>2000)return'';const x=new URL(u,location.href);return(x.protocol==='http:'||x.protocol==='https:')&&x.origin===location.origin?x.href:''}catch(e){return''}};
const opsSet=(id,k)=>{if(!id)return;OPS[id]=k;const ks=Object.keys(OPS);if(ks.length>100)ks.slice(0,ks.length-100).forEach(x=>delete OPS[x])};
const ci=c=>{const n=Number(c);return(c!==''&&c!=null&&Number.isInteger(n))?n:c};
const APPLIED={};const apk=k=>(S&&S.job)+'|'+k;
const uid=()=>Date.now().toString(36)+Math.random().toString(36).slice(2,10);
const pvKey=(l,c,i)=>l+'/'+c+'/'+i;
let islLast='';
function islStage(stage,seq,opId){const k=(opId||'')+'|'+stage;if(k===islLast)return;islLast=k;ISL.push({seq:seq||0,stage});if(ISL.length>200)ISL.splice(0,ISL.length-200);showToast(STAGE_TXT[stage]||String(stage).slice(0,40),stage==='complete'?2500:30000,stage==='error')}
function xpBar(seg){return(CTL.on&&CTL.expr)?'<div class=xp role=group aria-label="Change how this line is delivered">'+XI.map(([k,l])=>`<button type=button data-xp=${k}>${l}</button>`).join('')+((APPLIED[apk(pvKey(S&&S.lang,seg.chunk,seg.id))]||seg.expr_applied||seg.expression_applied)?'<button type=button data-xp-undo>Undo</button>':'')+'</div>':''}
function ctlErr(st,j,ra){const c=j&&j.code,d=j&&typeof j.detail==='string'?j.detail:'';
 if(st===409){if(c==='busy')return'Another change to this line is still running.';if(c==='nothing_to_undo')return'There is nothing to undo on this line.';return'This line was changed elsewhere. The latest version is loaded. Please try again.'}
 if(st===429)return c==='spend_cap'?'The spending limit for this job has been reached.':`Too many requests. Try again${ra?' in '+ra+' s':' shortly'}.`;
 if(st===501)return'This capability is not available on the server yet.';
 if(st===422)return'The request was not accepted'+(d?': '+d.slice(0,160):'.');
 return'The request failed (HTTP '+st+').'}
async function ctlPost(J,body){let r;try{r=await fetch(`/api/jobs/${E(J)}/actions`,{method:'POST',headers:hdr({'Content-Type':'application/json'}),body:JSON.stringify(body)})}catch(e){return{st:0,j:null}}
 let j=null;try{j=await r.json()}catch(e){}return{st:r.status,j,ra:r.headers.get('Retry-After')}}
async function ctlIntent(p,key,J,lang,chunk,seg){if(p.intent||!Number.isInteger(p.rev))return;try{const r=await fetch(`/api/jobs/${E(J)}/revisions?lang=${E(lang)}&chunk=${E(chunk)}&segment=${E(seg)}`,{headers:hdr()});if(!r.ok)return;const a=await r.json(),l=Array.isArray(a)?a:(a&&a.revisions)||[];const x=l.find(y=>y&&(y.revision|0)===p.rev),it=x&&x.expression&&x.expression.intent;if(XL[it]&&PV[key]===p){p.intent=it;xpRenderKey(key)}}catch(e){}}
async function ctlHead(J,lang,chunk,seg){try{const r=await fetch(`/api/jobs/${E(J)}/revisions?lang=${E(lang)}&chunk=${E(chunk)}&segment=${E(seg)}`,{headers:hdr()});if(!r.ok)return 0;const a=await r.json();if(a&&!Array.isArray(a)&&Number.isInteger(a.head_revision))return a.head_revision;const l=Array.isArray(a)?a:(a&&a.revisions)||[];let hd=0;l.forEach(x=>{if(x&&x.status!=='candidate'&&x.status!=='failed'&&(x.revision|0)>=hd)hd=x.revision|0});return hd}catch(e){return 0}}
/* ---- SSE ---- */
async function sseRead(body,cb,tick){const rd=body.getReader(),td=new TextDecoder();let buf='';
 for(;;){const{done,value}=await rd.read();if(done)break;if(tick)tick();buf+=td.decode(value,{stream:true}).replace(/\r\n/g,'\n');if(buf.length>262144&&buf.indexOf('\n\n')<0){try{rd.cancel()}catch(e){}throw Error('frame too large')}let i;
  while((i=buf.indexOf('\n\n'))>=0){const fr=buf.slice(0,i);buf=buf.slice(i+2);if(fr.length>262144){try{rd.cancel()}catch(e){}throw Error('frame too large')}let ev='message',data=[],id=null;
   fr.split('\n').forEach(l=>{if(!l||l[0]===':')return;const k=l.indexOf(':'),f=k<0?l:l.slice(0,k);let v=k<0?'':l.slice(k+1);if(v[0]===' ')v=v.slice(1);if(f==='event')ev=v;else if(f==='data')data.push(v);else if(f==='id')id=v});
   if(data.length){let d=null;try{d=JSON.parse(data.join('\n'))}catch(e){}if(d&&typeof d==='object')cb(ev,d,id)}}}}
const sleep=ms=>new Promise(res=>setTimeout(res,ms));
const opsActive=()=>CHAT.open||Object.keys(OPS).length>0||Object.values(PV).some(p=>p&&p.busy);
async function ctlPoll(J,my,n){let fails=0,done=0;const base=window.__CTL_POLL_MS||1500;
 while(done<n&&CTL.gen===my&&CTL.job===J){
  if(document.hidden){await new Promise(res=>{const f=()=>{if(!document.hidden){document.removeEventListener('visibilitychange',f);clearTimeout(t);res()}},t=setTimeout(()=>{document.removeEventListener('visibilitychange',f);res()},30000);document.addEventListener('visibilitychange',f)});continue}
  if(!opsActive()){await sleep(Math.max(500,base/2));continue}
  done++;let more=false;const ac=new AbortController(),to=setTimeout(()=>ac.abort(),10000);
  try{const r=await fetch(`/api/jobs/${E(J)}/events/poll?after=${CTL.seq}`,{headers:hdr(),signal:ac.signal});
   if(!r.ok)throw 0;const a=await r.json(),l=Array.isArray(a)?a:(a&&Array.isArray(a.events)?a.events:[]);
   l.forEach(ev=>{if(CTL.gen===my&&ev&&typeof ev==='object')ctlEvent('message',ev)});
   if(a&&a.reset===true&&CTL.gen===my){if(S&&S.lang)loadTranscript(S.lang,{force:true});if(Number.isInteger(a.last_seq)&&a.last_seq>CTL.seq)CTL.seq=a.last_seq}
   more=l.length>=100&&!(a&&a.reset===true);fails=0}catch(e){fails++}finally{clearTimeout(to)}
  if(CTL.gen!==my)return;
  await sleep(more?0:Math.min(15000,base*Math.pow(2,fails)))}}
async function ctlBaseline(J){const ac=new AbortController(),to=setTimeout(()=>ac.abort(),5000);try{const r=await fetch(`/api/jobs/${E(J)}/events/poll?after=2147483647`,{headers:hdr(),signal:ac.signal});if(r.ok){const a=await r.json();if(a&&Number.isInteger(a.last_seq)&&!CTL.seq)CTL.seq=a.last_seq}}catch(e){}finally{clearTimeout(to)}}
async function ctlStream(J){const my=++CTL.gen,base=window.__CTL_BACKOFF||500,silentMs=window.__CTL_SILENT_MS||5000;let back=base;
 while(CTL.gen===my&&CTL.job===J){const ac=new AbortController();CTL.ac=ac;let silent=false,wd=0;
  try{const h=hdr({Accept:'text/event-stream'});if(CTL.seq)h['Last-Event-ID']=String(CTL.seq);
   const arm=()=>{clearTimeout(wd);wd=setTimeout(()=>{silent=true;try{ac.abort()}catch(e){}},silentMs)};
   const t0=Date.now();let got=0;arm();const r=await fetch(`/api/jobs/${E(J)}/events`+(CTL.seq?`?after=${CTL.seq}`:''),{headers:h,signal:ac.signal});if(!r.ok||!r.body)throw 0;
   try{await sseRead(r.body,(n,d)=>{if(CTL.gen===my){got++;ctlEvent(n,d)}},arm)}finally{clearTimeout(wd);if(got>0&&Date.now()-t0>=5000)back=base}}catch(e){clearTimeout(wd);if(CTL.gen!==my)return}
  if(silent){await ctlPoll(J,my,20);back=base;continue}
  await new Promise(res=>setTimeout(res,back*(0.5+Math.random()*0.5)));back=Math.min(back*2,30000)}}
function ctlEvent(name,d){const t=(name&&name!=='message')?name:d.type,sq=typeof d.seq==='number'?d.seq:0;
 if(t==='reset'){CTL.seq=sq||0;if(S&&S.lang)loadTranscript(S.lang,{force:true});return}
 if(sq){if(sq<=CTL.seq)return;CTL.seq=sq}
 let key=OPS[d.operation_id],p=key&&PV[key];
 if(!p&&d.operation_id&&d.lang&&d.segment_id!=null&&d.chunk!=null&&((t==='op.queued'&&d.action_type==='change_expression')||(t==='op.audio_ready'&&d.candidate===true))){
  const k2=pvKey(d.lang,d.chunk,d.segment_id);if(PV[k2]&&!PV[k2].opId){/* a line-button preview is mid-POST */}else{p=PV[k2]=PV[k2]&&PV[k2].opId===d.operation_id?PV[k2]:{intent:'',busy:true,state:'queued',opId:d.operation_id,parent:Number.isInteger(d.head_revision)?d.head_revision:0};key=k2;opsSet(d.operation_id,k2)}}
 if(t==='op.queued')islStage('queued',sq,d.operation_id);
 else if(t==='op.stage'){islStage(d.stage,sq,d.operation_id);if(p&&d.stage!=='audio_ready'&&p.state!=='ready'){p.state=d.stage;xpRenderKey(key)}}
 else if(t==='op.audio_ready'){islStage('audio_ready',sq,d.operation_id);
  if(p&&d.candidate!==false){p.state='ready';p.busy=false;if(Number.isInteger(d.candidate_revision))p.rev=d.candidate_revision;else if(Number.isInteger(d.revision))p.rev=d.revision;if(Number.isInteger(d.head_revision))p.parent=d.head_revision;p.audio=mediaUrl(d.audio_url||d.preview_url)||p.audio||'';p.base=mediaUrl(d.base_audio_url)||p.base||'';xpRenderKey(key);if(!p.intent)ctlIntent(p,key,S.job,d.lang||S.lang,d.chunk,d.segment_id);if(CHAT.open&&innerWidth<900){chatShow(false);showToast('Preview ready. Compare it on the line.',4000)}}}
 else if(t==='op.lipsync_ready')islStage('lip_syncing',sq,d.operation_id);
 else if(t==='op.done'){islStage('complete',sq,d.operation_id);if(p&&p.applying){pvDel(key);if(p.undoing)delete APPLIED[apk(key)];else APPLIED[apk(key)]=true;ctlRefresh(key)}delete OPS[d.operation_id]}
 else if(t==='op.error'){delete OPS[d.operation_id];islStage('error',sq,d.operation_id);const m=typeof d.error==='string'?d.error.slice(0,160):'';if(p){p.state='failed';p.err=m||'The change could not be completed.';p.busy=false;xpRenderKey(key)}else showToast('That did not work.'+(m?' '+m:''),8000,true)}
 else if(t==='revision.committed'||t==='revision.undone'){const r=d.revision||{};if(t==='revision.undone'&&r.lang)delete APPLIED[apk(pvKey(r.lang,r.chunk_id,r.segment_id))];if(S&&r.lang&&r.lang===S.lang)loadTranscript(S.lang,{force:true})}}
async function ctlRefresh(key){if(!S)return;const[l,c]=key.split('/');if(l!==S.lang)return;const J=S.job;await loadTranscript(l,{force:true});if(S&&S.job===J)reloadPlayback(c,J,l)}
function ctlClose(){try{CHAT.ac&&CHAT.ac.abort()}catch(e){}CHAT.ac=null;CHAT.busy=false;try{$('chatsend').disabled=false}catch(e){}capCache.clear();capSig='';CTL.gen++;try{CTL.ac&&CTL.ac.abort()}catch(e){}CTL.job=null;CTL.seq=0;CTL.on=false;islLast='';Object.keys(PV).forEach(pvDel);Object.keys(APPLIED).forEach(k=>delete APPLIED[k]);Object.keys(PROBED).forEach(k=>delete PROBED[k]);Object.keys(OPS).forEach(k=>delete OPS[k]);const cb=$('chatbtn');if(cb)cb.hidden=true;chatShow(false);$('chatlog').textContent='';CHAT.hist=[]}
async function ctlCfg(){if(CTL.cfg)return CTL.cfg;for(let i=0;i<3;i++){try{const r=await fetch('/api/ctl/config',{headers:hdr()});if(r.ok){const c=await r.json();if(c&&c.enabled===true)CTL.cfg=c;return c}if(r.status<500)return null}catch(e){}await new Promise(res=>setTimeout(res,400*2**i))}return null}
async function ctlOpen(){if(!S||!S.job)return;const J=S.job;if(CTL.job===J&&(CTL.on||CTL.cfgP))return;if(CTL.job&&CTL.job!==J)ctlClose();CTL.job=J;
 if(!CTL.cfgP)CTL.cfgP=ctlCfg().finally(()=>{CTL.cfgP=null});
 const c=await CTL.cfgP;if(CTL.job!==J||!c||c.enabled!==true)return;
 await ctlBaseline(J);if(CTL.job!==J)return;
 const f=c.flags||{};CTL.on=true;CTL.expr=f.expression!==false;CTL.chat=f.chat!==false;$('chatbtn').hidden=!CTL.chat;
 if(CTL.expr&&S.lang&&S.lang!=='original'&&S.transcripts&&S.transcripts[S.lang])renderTranscriptRows();
 ctlStream(J)}
/* ---- per-line expression ---- */
function xpRenderKey(key){const[l,c,i]=key.split('/'),row=rowOf(i,c);if(row&&row.dataset.lang===l)xpRender(row,PV[key])}
const PROBED={};
function xpRevoke(p){if(p&&p.blobs){Object.values(p.blobs).forEach(u=>{try{URL.revokeObjectURL(u)}catch(e){}});p.blobs={}}}
function pvDel(k){const p=PV[k];if(p){if(p.opId)delete OPS[p.opId];try{p.au&&p.au.pause()}catch(e){}xpRevoke(p)}delete PV[k]}
async function ctlProbe(){if(!CTL.on||!CTL.expr||!S||!S.lang||S.lang==='original')return;const lang=S.lang,J=S.job,arr=S.transcripts[lang]||[];
 for(const sg of arr){if(sg.expr_applied||sg.ctl_head==null||sg.ctl_head<1)continue;const key=pvKey(lang,sg.chunk,sg.id),pk=J+key+'@'+sg.ctl_head;if(PROBED[pk])continue;if(Object.keys(PROBED).length>400)Object.keys(PROBED).forEach(k=>delete PROBED[k]);PROBED[pk]=1;
  try{const r=await fetch(`/api/jobs/${E(J)}/revisions?lang=${E(lang)}&chunk=${E(sg.chunk)}&segment=${E(sg.id)}`,{headers:hdr()});if(!r.ok||!S||S.job!==J)continue;const a=await r.json(),l=(a&&a.revisions)||[],hd=l.find(x=>x&&(x.revision|0)===sg.ctl_head);
   if(hd&&hd.status==='active'&&hd.m2_applied&&hd.expression&&hd.expression.provenance!=='undo_line'&&hd.expression.intent){APPLIED[apk(key)]=true;const row=rowOf(sg.id,sg.chunk);if(row&&S.lang===lang&&!row.querySelector('[data-xp-undo]')){const bar=row.querySelector('.xp');if(bar){const b=document.createElement('button');b.type='button';b.setAttribute('data-xp-undo','');b.textContent='Undo';bar.appendChild(b)}}}}catch(e){}}}
function xpRestore(){Object.keys(PV).forEach(xpRenderKey)}
function xpRender(row,p){const old=row.querySelector('.xpv');if(!p){if(old)old.remove();row.classList.remove('xpon');row.querySelectorAll('[data-xp],[data-xp-undo]').forEach(b=>b.disabled=false);return}
 row.classList.add('xpon');row.querySelectorAll('[data-xp],[data-xp-undo]').forEach(b=>{b.disabled=!!p.busy||p.state==='ready'||p.state==='confirm'});
 const n=old||document.createElement('div');n.className='xpv'+(p.state==='failed'?' err':'');n.textContent='';n.dataset.state=p.state;
 const s=document.createElement('div');s.className='xps';s.setAttribute('role','status');const L=XL[p.intent]||'New';
 const mk=(t,f,pressed)=>{const b=document.createElement('button');b.type='button';b.textContent=t;b.addEventListener('click',f);if(pressed!=null)b.setAttribute('aria-pressed',String(pressed));return b};
 const bar=document.createElement('div');bar.className='xpb';
 if(p.state==='ready'){s.textContent=`${L} version is ready. Compare it with the current line, then apply or discard.`;
  bar.appendChild(mk('A · Current',()=>xpPlay(row,p,'A'),p.playing==='A'));bar.appendChild(mk('B · '+L,()=>xpPlay(row,p,'B'),p.playing==='B'));
  bar.appendChild(mk('Apply',()=>xpApply(row,p),null)).classList.add('btn');bar.appendChild(mk('Discard',()=>xpDiscard(row,p),null));
  if(!p.base)bar.children[0].disabled=true}
 else if(p.state==='confirm'&&p.confirm){s.textContent=p.confirm.summary;bar.appendChild(mk('Confirm',()=>xpConfirm(row,p),null)).classList.add('btn');bar.appendChild(mk('Cancel',()=>xpDiscard(row,p),null))}
 else if(p.state==='applying'||p.state==='lip_syncing'){s.textContent=p.state==='lip_syncing'?'Applied. Lip-syncing this line…':(p.undoing?'Undoing…':'Applying…')}
 else if(p.state==='failed'){s.textContent=p.err||'This change could not be completed.';bar.appendChild(mk('Dismiss',()=>xpDiscard(row,p),null))}
 else s.textContent=`Preparing a ${L.toLowerCase()} version… ${STAGE_TXT[p.state]||''}`;
 n.appendChild(s);if(bar.children.length)n.appendChild(bar);if(!old)row.querySelector('.sbox').appendChild(n)}
function xpStop(p){p.tok=(p.tok||0)+1;if(p.au){try{p.au.pause()}catch(e){}clearTimeout(p.auT)}p.playing=''}
async function xpPlay(row,p,which){xpStop(p);const src=mediaUrl(which==='B'?p.audio:p.base);if(!src)return;
 const tok=(p.tok=(p.tok||0)+1),key=Object.keys(PV).find(x=>PV[x]===p);p.playing=which;row.dataset.xpPlaying=which;xpRender(row,p);
 let u=p.blobs&&p.blobs[src];
 if(!u){let nu=null;try{const r=await fetch(src,{headers:hdr()});if(!r.ok)throw 0;nu=URL.createObjectURL(await r.blob())}catch(e){if(p.tok===tok&&PV[key]===p){p.playing='';row.dataset.xpPlaying='';xpRender(row,p);showToast('The preview audio could not be loaded.',5000,true)}return}
  if(p.tok!==tok||!key||PV[key]!==p||!row.isConnected&&!rowOf(row.dataset.id,row.dataset.chunk)){try{URL.revokeObjectURL(nu)}catch(e){}return}
  (p.blobs=p.blobs||{})[src]=nu;u=nu}
 if(p.tok!==tok||PV[key]!==p)return;
 const a=p.au||(p.au=new Audio());a.src=u;a.currentTime=0;
 const ms=Math.max(500,((parseFloat(row.dataset.end)-parseFloat(row.dataset.start))||4)*1000+250);clearTimeout(p.auT);p.auT=setTimeout(()=>{xpStop(p);row.dataset.xpPlaying='';xpRender(row,p)},ms);
 a.play().catch(()=>{})}
function xpDiscard(row,p){xpStop(p);Object.values(p.blobs||{}).forEach(u=>{try{URL.revokeObjectURL(u)}catch(e){}});const k=Object.keys(PV).find(x=>PV[x]===p);if(k)pvDel(k);row.dataset.xpPlaying='';xpRender(row,null)}
async function xpRun(row,intent){if(!S||S.mut)return;const id=row.dataset.id,chunk=row.dataset.chunk,lang=S.lang,J=S.job,key=pvKey(lang,chunk,id);
 if(PV[key]&&PV[key].busy)return;if(PV[key]){xpStop(PV[key]);xpRevoke(PV[key])}
 const p=PV[key]={intent,busy:true,state:'queued'};xpRender(row,p);
 const head=await ctlHead(J,lang,chunk,id);if(!S||S.job!==J||PV[key]!==p)return;p.parent=head;
 const r=await ctlPost(J,{expected_revision:head,idempotency_key:uid(),type:'change_expression',args:{lang,chunk:ci(chunk),segment_id:id,intent},preview_only:true});
 if(!S||S.job!==J||PV[key]!==p)return;
 if(r.st===202||r.st===200){const j=r.j||{},op=j.operation||{};if(!op.operation_id){pvDel(key);xpRender(row,null);showToast('The server did not start the change.',6000,true);return}
  opsSet(op.operation_id,key);p.opId=op.operation_id;if(j.revision&&Number.isInteger(j.revision.revision))p.rev=j.revision.revision;if(Number.isInteger(j.head_revision))p.parent=j.head_revision;p.state=op.stage||'queued';
  islStage(p.state,0,op.operation_id);
  if(op.audio_ready){p.state='ready';p.audio=mediaUrl((j.revision&&j.revision.audio_url)||op.audio_url);p.base=mediaUrl(j.revision&&j.revision.base_audio_url)}
  p.busy=false;if(p.state==='ready')p.busy=false;else p.busy=true;xpRender(row,p);return}
 pvDel(key);xpRender(row,null);showToast(ctlErr(r.st,r.j,r.ra),7000,true);if(r.st===409)loadTranscript(lang,{force:true})}
async function xpSubmit(row,p,key,body,isUndo){const J=S.job,lang=S.lang;
 const r=await ctlPost(J,body);if(!S||S.job!==J||PV[key]!==p)return;const j=r.j||{};
 if(r.st===200&&(j.status==='needs_confirmation'||j.status==='fork_proposal')&&j.confirmation_token){p.state='confirm';p.busy=false;p.confirm={token:j.confirmation_token,body,summary:(()=>{const sm=String((j.proposal&&j.proposal.summary)||'');return(/\s/.test(sm)&&sm.length>12)?sm.slice(0,300):(isUndo?'Undo the last change to this line?':'Apply this version to the line?')})(),isUndo};xpRender(row,p);return}
 if(r.st===202||r.st===200){p.busy=true;p.applying=true;p.undoing=!!isUndo;p.state=isUndo?'applying':'applying';const op=(j&&j.operation)||{};if(op.operation_id)opsSet(op.operation_id,key);islStage(op.stage&&op.stage!=='queued'?op.stage:'queued',0,op.operation_id);xpRender(row,p);return}
 pvDel(key);xpRender(row,null);showToast(ctlErr(r.st,r.j,r.ra),7000,true);if(r.st===409)loadTranscript(lang,{force:true})}
async function xpApply(row,p){if(!S||p.busy&&p.state!=='ready')return;const id=row.dataset.id,chunk=row.dataset.chunk,lang=S.lang,J=S.job,key=pvKey(lang,chunk,id);xpStop(p);
 if(!Number.isInteger(p.rev)){showToast('This preview is not ready to apply yet.',5000,true);return}
 xpRevoke(p);p.busy=true;p.state='applying';xpRender(row,p);
 await ctlIntent(p,key,J,lang,chunk,id);
 if(!p.intent){p.busy=false;p.state='ready';xpRender(row,p);showToast('Could not tell which delivery this preview is. Discard it and try again.',6000,true);return}
 const head=await ctlHead(J,lang,chunk,id);if(!S||S.job!==J||PV[key]!==p)return;
 await xpSubmit(row,p,key,{expected_revision:head,idempotency_key:uid(),type:'change_expression',args:{lang,chunk:ci(chunk),segment_id:id,intent:p.intent,apply_candidate:String(p.rev)},preview_only:false},false)}
async function xpConfirm(row,p){const key=Object.keys(PV).find(x=>PV[x]===p);if(!key||!p.confirm||!S)return;const c=p.confirm;p.confirm=null;p.busy=true;p.state='applying';xpRender(row,p);
 await xpSubmit(row,p,key,Object.assign({},c.body,{idempotency_key:uid(),confirmation_token:c.token}),c.isUndo)}
async function xpUndo(row){if(!S||S.mut)return;const id=row.dataset.id,chunk=row.dataset.chunk,lang=S.lang,J=S.job,key=pvKey(lang,chunk,id);
 if(PV[key]&&PV[key].busy)return;const p=PV[key]={intent:'',busy:true,state:'applying',undoing:true};xpRender(row,p);
 const head=await ctlHead(J,lang,chunk,id);if(!S||S.job!==J||PV[key]!==p)return;
 await xpSubmit(row,p,key,{expected_revision:head,idempotency_key:uid(),type:'undo_line',args:{lang,chunk:ci(chunk),segment_id:id},preview_only:false},true)}
segsBox.addEventListener('click',e=>{const u=e.target.closest('[data-xp-undo]');if(u){const rw=u.closest('.seg-row');if(rw&&S&&rw.dataset.lang===S.lang){e.stopPropagation();xpUndo(rw)}return}const b=e.target.closest('[data-xp]');if(!b)return;const row=b.closest('.seg-row');if(!row||!S||row.dataset.lang!==S.lang)return;e.stopPropagation();xpRun(row,b.dataset.xp)});
/* ---- chat slide-over ---- */
const CHAT={hist:[],busy:false,open:false,ac:null};
function chatShow(on){if(on&&!CHAT.open){CHAT.prev=document.activeElement}const was=CHAT.open;CHAT.open=on;document.body.classList.toggle('chatopen',on);const c=$('chat');c.classList.toggle('on',on);c.setAttribute('aria-hidden',String(!on));try{document.dispatchEvent(new CustomEvent('subreflow',{detail:on?{width:Math.min(400,innerWidth)}:{}}))}catch(e){}if(on){chatChips();setTimeout(()=>{if(CHAT.open)$('chatin').focus()},60)}else if(was){const pv=CHAT.prev;CHAT.prev=null;setTimeout(()=>{const t=(pv&&pv.isConnected&&pv!==document.body&&!pv.closest('#chat'))?pv:$('chatbtn');if(t&&!t.hidden)try{t.focus({preventScroll:true})}catch(e){}},0)}}
function chatChips(){const b=$('chatchips');if(b.childElementCount)return;
 [...XI.map(([k,l])=>[l,()=>chatLine(r=>xpRun(r,k))]),['Undo line',()=>chatLine(r=>{if(S.mut)return;if(r.querySelector('.sbox>.sact [data-act=undo]'))editUndo(r.dataset.id,r.dataset.chunk);else showToast('There is nothing to undo on this line.')})]].forEach(([t,f])=>{const x=document.createElement('button');x.type='button';x.textContent=t;x.addEventListener('click',f);b.appendChild(x)})}
function chatLine(fn){if(!S||S.lang==='original'){showToast('Choose a dubbed language first.');return}const r=curRow();if(!r){showToast('Play or select a transcript line first.');return}fn(r)}
function chatAdd(role,text,cls){const m=document.createElement('div');m.className='cm '+role+(cls?' '+cls:'');m.textContent=String(text);const l=$('chatlog');l.appendChild(m);l.scrollTop=l.scrollHeight;return m}
async function chatSend(msg){if(CHAT.busy||!S||!CTL.on)return;msg=msg.trim().slice(0,1000);if(!msg)return;const J=S.job;CHAT.busy=true;$('chatsend').disabled=true;const ac=CHAT.ac=new AbortController();chatAdd('user',msg);
 const r0=curRow(),ctx={lang:S.lang,chunk:r0?ci(r0.dataset.chunk):null,segment_id:r0?r0.dataset.id:null,line_index:r0?[...segsBox.querySelectorAll('.seg-row')].indexOf(r0)+1:null,playhead_ms:Math.round((v.currentTime||0)*1000),subtitle_mode:S.captions?'dub':'off'};
 const bot=chatAdd('bot','');bot.hidden=true;let txt='',failed=false;
 let wd=0,timedOut=false;const arm=()=>{clearTimeout(wd);wd=setTimeout(()=>{timedOut=true;try{ac.abort()}catch(e){}},window.__CHAT_TIMEOUT_MS||45000)};arm();
 try{const r=await fetch(`/api/jobs/${E(J)}/chat`,{method:'POST',signal:ac.signal,headers:hdr({'Content-Type':'application/json',Accept:'text/event-stream'}),body:JSON.stringify({message:msg,history:CHAT.hist.slice(-10),context:ctx})});
  if(!r.ok||!r.body)throw 0;
  await sseRead(r.body,(n,d)=>{if(!S||S.job!==J)return;
   if(n==='text'){txt=(txt+String(d.delta==null?'':d.delta)).slice(0,20000);bot.hidden=false;bot.textContent=txt;$('chatlog').scrollTop=1e9}
   else if(n==='proposal')chatProposal(d);
   else if(n==='client_action')chatClient(d);
   else if(n==='action_result'){chatAdd('sys','Started — progress shows in the island.');if(d.operation_id&&!(d.operation_id in OPS))opsSet(d.operation_id,'')}
   else if(n==='error'){failed=true;chatAdd('sys',d.code==='planner_unavailable'?'The assistant is unavailable right now. The line controls still work.':'The assistant could not answer'+(d.detail?': '+String(d.detail).slice(0,160):'.'),'err')}},arm)}
 catch(e){if(CHAT.ac!==ac)return;failed=true;chatAdd('sys',timedOut?'The assistant took too long to answer. Try again; the line controls still work.':'The assistant is unavailable right now. The line controls (Angrier, Sadder, Calmer, Excited, Edit, Undo) still work.','err')}finally{clearTimeout(wd)}
 if(!txt)bot.remove();CHAT.hist.push({role:'user',text:msg});if(txt)CHAT.hist.push({role:'assistant',text:txt.slice(0,1000)});CHAT.hist=CHAT.hist.slice(-10);
 if(CHAT.ac===ac){CHAT.ac=null;CHAT.busy=false;$('chatsend').disabled=false}}
document.addEventListener('dub:subtitles',e=>{const m=e.detail&&e.detail.mode;if(['off','original','dub','dual'].includes(m)&&FX.setSubMode)FX.setSubMode(m)});
function chatClient(d){const a=d.args||{};
 if(d.tool==='switch_language'&&typeof a.lang==='string'&&SAFE.test(a.lang)&&S&&S.data&&S.data.langs[a.lang]){setTrack(a.lang);chatAdd('sys','Switched to '+LN(a.lang)+'.')}
 else if(d.tool==='subtitles'&&['off','original','dub','dual'].includes(a.mode)){document.dispatchEvent(new CustomEvent('dub:subtitles',{detail:{mode:a.mode}}));chatAdd('sys','Subtitles: '+a.mode+'.')}
 else if(d.tool==='inspect')chatAdd('sys','Looked at the current job.');
 else chatAdd('sys','That action is not available here.')}
function chatProposal(pr){const J=S.job,c=document.createElement('div');c.className='cp';c.setAttribute('role','group');c.setAttribute('aria-label','Confirmation needed');
 const t=document.createElement('div');t.textContent=String(pr.summary||'Proposed change').slice(0,400);c.appendChild(t);
 if(pr.requires_confirmation){const b=document.createElement('div');b.className='cpb';
  const ok=document.createElement('button');ok.type='button';ok.className='btn';ok.textContent='Confirm';const no=document.createElement('button');no.type='button';no.textContent='Cancel';
  no.onclick=()=>{c.remove();chatAdd('sys','Cancelled.')};
  ok.onclick=async()=>{ok.disabled=no.disabled=true;
   const r=await ctlPost(J,{expected_revision:pr.expected_revision|0,idempotency_key:uid(),type:String(pr.tool),args:pr.args||{},preview_only:false,...(pr.confirmation_token?{confirmation_token:pr.confirmation_token}:{})});
   if(!S||S.job!==J)return;const j=r.j||{};
   if((r.st===200)&&(j.status==='needs_confirmation'||j.status==='fork_proposal')){pr.confirmation_token=j.confirmation_token;const pp=j.proposal||{};t.textContent=String(pp.summary||pr.summary||'Confirm to continue').slice(0,400);
    if(j.status==='fork_proposal'&&j.fork){t.textContent+=' This creates a new job with a modified voice plan; the current job is unchanged.'}ok.disabled=no.disabled=false;return}
   if(r.st===202||r.st===200){const op=j.operation||{};if(op.operation_id)opsSet(op.operation_id,'');c.remove();chatAdd('sys','Confirmed — progress shows in the island.');return}
   ok.disabled=no.disabled=false;chatAdd('sys',ctlErr(r.st,j,r.ra),'err');if(r.st===409||r.st===422){c.remove()}};
  b.appendChild(ok);b.appendChild(no);c.appendChild(b)}
 $('chatlog').appendChild(c);$('chatlog').scrollTop=1e9}
$('lipbtn').addEventListener('click',lipToggle);
$('chatbtn').addEventListener('click',()=>chatShow(!CHAT.open));$('chatx').addEventListener('click',()=>chatShow(false));
$('chatform').addEventListener('submit',e=>{e.preventDefault();const i=$('chatin'),m=i.value;i.value='';chatSend(m)});
$('chatin').addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey&&!e.isComposing){e.preventDefault();$('chatform').requestSubmit()}});
$('chat').addEventListener('keydown',e=>{if(e.key==='Escape'){e.stopPropagation();chatShow(false)}else if(e.key==='Tab'&&innerWidth<900){const f=[...$('chat').querySelectorAll('button,textarea,[href],input,select,[tabindex]:not([tabindex="-1"])')].filter(x=>!x.disabled&&!x.hidden&&x.offsetParent!==null);if(!f.length)return;const a=f[0],z=f[f.length-1];if(e.shiftKey&&document.activeElement===a){e.preventDefault();z.focus()}else if(!e.shiftKey&&document.activeElement===z){e.preventDefault();a.focus()}}});
document.addEventListener('keydown',e=>{if((e.ctrlKey||e.metaKey)&&!e.altKey&&e.key.toLowerCase()==='k'&&CTL.chat&&S&&document.body.dataset.view==='studio'){e.preventDefault();chatShow(!CHAT.open)}});
init();
})();
