/* FX: presentation layer (streaming-player conventions). It never touches playback state, the API or the DOM hooks the logic owns;
   everything it renders from live data goes through textContent. */
window.FX=(()=>{
const $=id=>document.getElementById(id);
const RM=matchMedia('(prefers-reduced-motion: reduce)'),rm=()=>RM.matches;
const v=$('v'),stage=$('stage'),ctl=stage.querySelector('.ctlwrap'),deck=$('deck'),hdr=document.querySelector('header.top'),stagein=$('stagein');
const SVG='<svg class=i viewBox="0 0 24 24" aria-hidden=true ';
const ICON={
 play:SVG+'fill=currentColor><path d="M8 5v14l12-7z"/></svg>',
 pause:SVG+'fill=currentColor><path d="M7 5h4v14H7zM13 5h4v14h-4z"/></svg>',
 volume:SVG+'fill=none stroke=currentColor stroke-width=1.8 stroke-linecap=round stroke-linejoin=round><path d="M4 9.5h3.2L12 5v14l-4.8-4.5H4z"/><path d="M15.5 9.3a3.6 3.6 0 0 1 0 5.4M18 6.8a7 7 0 0 1 0 10.4"/></svg>',
 muted:SVG+'fill=none stroke=currentColor stroke-width=1.8 stroke-linecap=round stroke-linejoin=round><path d="M4 9.5h3.2L12 5v14l-4.8-4.5H4z"/><path d="M16 9.5l5 5M21 9.5l-5 5"/></svg>',
 back:SVG+'fill=none stroke=currentColor stroke-width=1.8 stroke-linecap=round stroke-linejoin=round><path d="M4 11a8 8 0 1 0 2.4-5.7"/><path d="M4 4v5h5"/></svg>',
 fwd:SVG+'fill=none stroke=currentColor stroke-width=1.8 stroke-linecap=round stroke-linejoin=round><path d="M20 11a8 8 0 1 1-2.4-5.7"/><path d="M20 4v5h-5"/></svg>',
 check:SVG+'fill=none stroke=currentColor stroke-width=2.2 stroke-linecap=round stroke-linejoin=round><path d="M5 12.5l4.5 4.5L19 7.5"/></svg>',
 alert:SVG+'fill=none stroke=currentColor stroke-width=2 stroke-linecap=round><path d="M12 5v9M12 18.5v.5"/></svg>'};
const ic=k=>ICON[k]||'';
const F={src:''};
/* ---- view choreography ---- */
let viewT=0,hinted=false;
function view(n){const st=$('studio');clearTimeout(viewT);st.classList.remove('enter');
 if(n==='studio'){if(!rm()){void st.offsetWidth;st.classList.add('enter');viewT=setTimeout(()=>st.classList.remove('enter'),1200)}heroStop();capStop();stopPreviews(true);closePanel();stage.classList.remove('woke');wake()}
 else{reset();heroStart();restorePreviews();$('hdtitle').textContent='';document.body.classList.remove('hidectl');closeMore(false)}}
/* ---- landing media: hero clip + demo previews (registry; unloaded in the studio) ---- */
const herov=$('herov');let heroSet=false;
function heroStart(){if(rm()||!herov||document.hidden||document.body.dataset.view!=='landing')return;if(!heroSet){heroSet=true;herov.preload='auto';herov.src='/media/879b2e87a92a/original/0.mp4#t=12';herov.addEventListener('playing',()=>herov.classList.add('on'));herov.addEventListener('error',()=>herov.classList.remove('on'))}
 const p=herov.play();if(p&&p.catch)p.catch(()=>{})}
function heroStop(){if(herov&&!herov.paused)herov.pause()}
const PV=[];
document.querySelectorAll('.demo[data-pv]').forEach(d=>{const pv=d.querySelector('.pv');if(!pv)return;const id=d.dataset.pv;if(!/^[0-9a-f]{12}$/.test(id))return;
 const o={d,pv,src:'/media/'+id+'/original/0.mp4#t=6',t:0};PV.push(o);
 pv.preload='metadata';pv.style.opacity='0';pv.addEventListener('loadeddata',()=>{pv.style.opacity='1'});pv.addEventListener('error',()=>{pv.style.opacity='0'});
 d.addEventListener('mouseenter',()=>{if(rm()||document.body.dataset.view!=='landing')return;o.seen=true;if(!pv.getAttribute('src'))pv.src=o.src;clearTimeout(o.t);o.t=setTimeout(()=>{if(document.body.dataset.view!=='landing'||!pv.getAttribute('src'))return;const p=pv.play();if(p&&p.catch)p.catch(()=>{})},400)});
 d.addEventListener('focus',()=>{o.seen=true;if(document.body.dataset.view==='landing'&&!pv.getAttribute('src'))pv.src=o.src});
 d.addEventListener('mouseleave',()=>{clearTimeout(o.t);if(!pv.paused)pv.pause()});
 d.addEventListener('pointerdown',()=>{clearTimeout(o.t);if(!pv.paused)pv.pause()})});
function stopPreviews(unload){PV.forEach(o=>{clearTimeout(o.t);try{if(!o.pv.paused)o.pv.pause();if(unload&&o.pv.getAttribute('src')){o.pv.removeAttribute('src');o.pv.load();o.pv.style.opacity='0'}}catch(e){}})}
function restorePreviews(){}
let capQ=[],capBusy=false,capv=null;
function capture(o){if(o.cap||o.capq)return;o.capq=true;capQ.push(o);capNext()}
let capAC=null,capGen=0,capCur=null;
function capRestore(o){if(o.oldcap&&!o.cap){o.cap=o.oldcap;o.oldcap=null;o.capCss=o.oldCss}}
function capStop(){capGen++;PV.forEach(capRestore);if(capAC){capAC.abort();capAC=null}capBusy=false;if(capCur){capCur.capq=false;capCur=null}capQ.forEach(q=>{q.capq=false});capQ=[];if(capv){try{capv.removeAttribute('src');capv.load()}catch(e){}}}
function capNext(){if(capBusy||!capQ.length||document.hidden||document.body.dataset.view!=='landing')return;const o=capQ.shift();capBusy=true;capCur=o;
 if(!capv){capv=document.createElement('video');capv.muted=true;capv.preload='metadata';capv.playsInline=true;capv.setAttribute('aria-hidden','true');capv.style.cssText='position:absolute;width:1px;height:1px;opacity:0;pointer-events:none';document.body.appendChild(capv)}
 const ac=new AbortController(),g=capGen;capAC=ac;const done=fail=>{if(g!==capGen)return;ac.abort();capAC=null;capCur=null;try{capv.removeAttribute('src');capv.load()}catch(e){}capBusy=false;o.capq=false;if(fail===true&&!o.cap&&!o.retried){o.retried=true;setTimeout(()=>{if(o.seen&&document.body.dataset.view==='landing')capture(o)},1500)}else capRestore(o);if(o.cap)o.retried=false;capNext()};
 capv.addEventListener('seeked',()=>{try{if(capv.videoWidth){const c=document.createElement('canvas');const r=o.d.getBoundingClientRect();const dp=Math.min(window.devicePixelRatio||1,2),W=Math.max(320,Math.round(r.width*dp)),H=Math.round(W*9/16);o.capCss=r.width;c.width=W;c.height=H;c.className='pcap';c.setAttribute('aria-hidden','true');const g=c.getContext('2d');const sc=Math.max(W/capv.videoWidth,H/capv.videoHeight),w=capv.videoWidth*sc,h=capv.videoHeight*sc;g.drawImage(capv,(W-w)/2,(H-h)/2,w,h);o.d.insertBefore(c,o.pv);if(o.oldcap){o.oldcap.remove();o.oldcap=null}o.cap=c}}catch(e){}done(!o.cap)},{signal:ac.signal,once:true});
 capv.addEventListener('error',()=>done(true),{signal:ac.signal,once:true});setTimeout(()=>{if(capBusy&&g===capGen&&!ac.signal.aborted)done(true)},8000);
 capv.src=o.src}
if('IntersectionObserver' in window){const io=new IntersectionObserver(es=>{es.forEach(x=>{const o=PV.find(q=>q.d===x.target);if(!o)return;o.seen=x.isIntersecting;if(x.isIntersecting&&document.body.dataset.view==='landing')capture(o)})},{threshold:.25});PV.forEach(o=>io.observe(o.d))}else{PV.forEach(capture)}
document.addEventListener('visibilitychange',()=>{if(document.hidden){heroStop();capStop();stopPreviews(false)}else if(document.body.dataset.view==='landing'){heroStart();PV.forEach(o=>{if(o.seen&&!o.cap)capture(o)});capNext()}});
let capRT=null;addEventListener('resize',()=>{clearTimeout(capRT);capRT=setTimeout(()=>{if(document.hidden||document.body.dataset.view!=='landing')return;PV.forEach(o=>{if(!o.seen||!o.cap||o.capq||!o.capCss)return;if(o.d.getBoundingClientRect().width>o.capCss*1.3){o.oldcap=o.cap;o.oldCss=o.capCss;o.cap=null;capture(o)}})},400)});
$('heroplay').addEventListener('click',()=>{const d=document.querySelector('.demo[data-demo="2"]');if(d&&!d.disabled)d.click()});
/* "Dub it" always looks live: with an invalid link the capture layer explains instead of a dead button */
let noteT=0;
$('dubcap').addEventListener('click',()=>{const f=$('form');const why=(F.formWhy&&F.formWhy())||['url','Paste a YouTube link first.'];const el=$(why[0]);$('dubnote').textContent=why[1];$('dubnote').hidden=false;f.classList.add('warn');
 if(why[0]==='consent'){const l=el.closest('label');if(l){l.style.outline='1px solid var(--warn)';l.style.outlineOffset='4px';l.style.borderRadius='6px';setTimeout(()=>{l.style.outline=''},2200)}}
 try{el.focus({preventScroll:true})}catch(e){}clearTimeout(noteT);noteT=setTimeout(()=>{f.classList.remove('warn');$('dubnote').hidden=true},2200)});
$('url').addEventListener('input',()=>{$('dubnote').hidden=true;$('form').classList.remove('warn')});
$('url').addEventListener('keydown',e=>{if(e.key!=='Enter'||e.isComposing||e.keyCode===229)return;e.preventDefault();if($('dub').disabled)$('dubcap').click();else $('form').requestSubmit()});
/* header overflow (mobile) */
function moreOpen(){$('hmore').classList.add('on');$('morebtn').setAttribute('aria-expanded','true');const f=$('hmore').querySelector('button:not([hidden])');if(f)f.focus({preventScroll:true})}
function closeMore(restore){const m=$('hmore');if(!m.classList.contains('on'))return;m.classList.remove('on');$('morebtn').setAttribute('aria-expanded','false');if(restore)$('morebtn').focus({preventScroll:true})}
$('morebtn').addEventListener('click',()=>{if($('hmore').classList.contains('on'))closeMore(true);else moreOpen()});
document.addEventListener('click',e=>{if(!$('hmore').contains(e.target)&&!$('morebtn').contains(e.target))closeMore(false)});
document.addEventListener('keydown',e=>{if(e.key==='Escape'&&$('hmore').classList.contains('on')){e.preventDefault();e.stopImmediatePropagation();closeMore(true)}},true);
$('hmore').addEventListener('focusout',e=>{const to=e.relatedTarget;if(to&&!$('hmore').contains(to)&&to!==$('morebtn'))closeMore(false)});
/* ---- auto-hiding controls (+ header, island). Never while focus is inside the controls, the header or the panel. ---- */
let hideT=0,panelOpen=false,kbT=0,touchMode=false,eat=null,eatT=0;
const INTER='button,a[href],input,select,textarea,summary,[role=button],[role=radio],[role=tab]';
function eatClear(){eat=null;clearTimeout(eatT)}
addEventListener('keydown',()=>{kbT=Date.now()},true);addEventListener('pointerdown',()=>{kbT=0},true);
function kbFocus(){const a=document.activeElement;if(!a||!(ctl.contains(a)||hdr.contains(a)))return false;let fv=false;try{fv=a.matches(':focus-visible')}catch(e){fv=true}return fv&&(Date.now()-kbT)<6000}
function hide(){if(document.body.dataset.view!=='studio')return;if(v.paused||panelOpen||$('ov').style.display==='grid'||!$('lchoice').hidden)return;
 if((!touchMode&&(deck.matches(':hover')||hdr.matches(':hover')))||kbFocus()||$('audpanel').matches(':focus-within')||$('hmore').classList.contains('on')){clearTimeout(hideT);hideT=setTimeout(hide,1200);return}
 stage.classList.add('hidectl');document.body.classList.add('hidectl')}
function wake(){stage.classList.remove('hidectl');document.body.classList.remove('hidectl');clearTimeout(hideT);hideT=setTimeout(hide,touchMode?3000:2800)}
function woke(){stage.classList.add('woke')}
stage.addEventListener('pointerdown',e=>{touchMode=e.pointerType==='touch';eatClear();if(touchMode&&stage.classList.contains('hidectl')&&!(e.target.closest&&e.target.closest(INTER))){eat={pid:e.pointerId,t:e.target};eatT=setTimeout(eatClear,400)}},{capture:true,passive:true});
['pointercancel','lostpointercapture'].forEach(n=>stage.addEventListener(n,eatClear,{capture:true,passive:true}));
stage.addEventListener('click',e=>{if(!eat)return;const g=eat;eatClear();if(e.target!==g.t||(e.pointerId!==undefined&&e.pointerId!==g.pid)||(e.target.closest&&e.target.closest(INTER)))return;e.stopPropagation();e.preventDefault()},true);
['mousemove','pointerdown','touchstart'].forEach(e=>stage.addEventListener(e,ev=>{if(e==='mousemove'&&touchMode)return;woke();wake()},{passive:true}));
document.addEventListener('mousemove',()=>{if(document.body.classList.contains('hidectl'))wake()},{passive:true});
stage.addEventListener('mouseleave',()=>{clearTimeout(hideT);hideT=setTimeout(hide,900)});
ctl.addEventListener('focusin',wake);hdr.addEventListener('focusin',wake);
v.addEventListener('error',()=>{clearTimeout(hideT);stage.classList.remove('hidectl');document.body.classList.remove('hidectl')});
v.addEventListener('pause',()=>{setTimeout(()=>{if(v.paused&&stage.classList.contains('idle')){stage.classList.remove('hidectl');document.body.classList.remove('hidectl');clearTimeout(hideT)}},350)});v.addEventListener('play',woke);
document.addEventListener('keydown',()=>{if(document.body.dataset.view==='studio'){woke();wake()}},{capture:true});
/* ---- centre feedback ---- */
let fbT=0;
function feedback(kind){if(rm())return;const f=$('feedback');f.innerHTML=ic(kind==='back'?'back':kind==='fwd'?'fwd':kind==='pause'?'pause':'play')+(kind==='back'||kind==='fwd'?'<span class=lb>10 s</span>':'');f.classList.remove('go');void f.offsetWidth;f.classList.add('go');clearTimeout(fbT);fbT=setTimeout(()=>f.classList.remove('go'),700)}
/* ---- buffering ring (no blur; the blurred overlay is for errors/waits the logic owns) ---- */
let bufT=0,bufL=0;const ring=$('buffring'),blbl=$('buflbl');
function bufOn(){clearTimeout(bufT);bufT=setTimeout(()=>{if(v.paused||v.ended||v.readyState>=3)return;ring.classList.add('on');clearTimeout(bufL);bufL=setTimeout(()=>{blbl.textContent='Loading…';blbl.classList.add('on')},3000)},400)}
function bufOff(){clearTimeout(bufT);clearTimeout(bufL);ring.classList.remove('on');blbl.classList.remove('on')}
v.addEventListener('waiting',bufOn);v.addEventListener('stalled',bufOn);v.addEventListener('playing',bufOff);v.addEventListener('pause',bufOff);v.addEventListener('canplay',bufOff);v.addEventListener('emptied',bufOff);
/* ---- Audio & subtitles panel: a modal dialog mirroring the #tracks chips and the caption toggle ---- */
const panel=$('audpanel'),audbtn=$('audbtn');let autoT=0,panelSig='',lastFocus=null,panelGen=0,pnoteT=0;
function note(msg){const t=$('toast');t.textContent=msg;t.classList.remove('e');t.classList.add('on');clearTimeout(pnoteT);pnoteT=setTimeout(()=>t.classList.remove('on'),3000)}
function stateOf(t){const d=t.querySelector('.dot');if(!d)return['',''];if(d.classList.contains('lips'))return['lips','Lips matched'];if(d.classList.contains('voice'))return['voice','Voice ready'];if(d.classList.contains('work'))return['work','Preparing'];return['','Not ready']}
function panelData(){const cap=$('capbtn'),on=cap.getAttribute('aria-pressed')==='true',tl=$('tlang'),ln=tl&&tl.selectedIndex>=0?tl.options[tl.selectedIndex].textContent:'';
 const rows=[...document.querySelectorAll('#tracks .tt')].map(t=>{const [k,label]=stateOf(t),n=t.querySelector('.n');let name=(t.textContent||'').replace(/\s*\d+\/\d+\s*$/,'').trim();if(t.dataset.trk==='original'&&F.src)name='Original ('+F.src+')';return{key:t.dataset.trk,name,on:t.getAttribute('aria-pressed')==='true',k,st:t.dataset.trk==='original'?'':label+(n&&k!=='lips'?' · '+n.textContent:'')}});
 return{rows,on,ln,mode:F.subModeName?F.subModeName():'dub',al:F.alData?F.alData().addT:{},open:F.alData?F.alData().open:false,wait:F.alData?F.alData().wait:'',aon:F.alData?F.alData().on:false}}
function buildPanel(force){
 const data=panelData(),sig=JSON.stringify(data);if(!force&&sig===panelSig)return;panelSig=sig;panelGen=jobGen;const gen=jobGen,keys=data.rows.map(r=>r.key);
 const fk=document.activeElement&&panel.contains(document.activeElement)?document.activeElement.dataset.key:null;
 const al=$('audlist'),sl=$('sublist');al.textContent='';sl.textContent='';
 const mkRow=(key,name,checked,st,k,act)=>{const b=document.createElement('button');b.type='button';b.className='ae';b.setAttribute('role','radio');b.setAttribute('aria-checked',checked?'true':'false');b.dataset.key=key;b.tabIndex=checked?0:-1;
  const ck=document.createElement('span');ck.className='ck';ck.innerHTML=ic('check');b.appendChild(ck);const nm=document.createElement('span');nm.className='nm';nm.textContent=name;b.appendChild(nm);
  if(st){const s=document.createElement('span');s.className='st '+k;s.textContent=st;b.appendChild(s)}b.addEventListener('click',act);return b};
 data.rows.forEach(r=>al.appendChild(mkRow('a:'+r.key,r.name,r.on,r.st,r.k,()=>{if(gen!==jobGen||!keys.includes(r.key)||!document.querySelector('#tracks .tt[data-trk="'+CSS.escape(r.key)+'"]')){note('That option is no longer available.');return}const res=F.setTrack?F.setTrack(r.key):false;if(res===true||res==='deferred')closePanel();else{note('Could not switch to '+r.name+'.');buildPanel(true)}})));
 const cap=$('capbtn');const capAct=m=>()=>{if(gen!==jobGen){note('That option is no longer available.');return}const ok=F.setSubMode?F.setSubMode(m):false;if(!ok){buildPanel(true);return}closePanel()};
 sl.appendChild(mkRow('s:off','Off',!data.on,'','',capAct('off')));if(data.ln){sl.appendChild(mkRow('s:on',data.ln,data.on&&data.mode==='dub','','',capAct('dub')));sl.appendChild(mkRow('s:orig','Original language',data.on&&data.mode==='original','','',capAct('original')));sl.appendChild(mkRow('s:dual','Both: original + '+data.ln,data.on&&data.mode==='dual','','',capAct('dual')))}
 buildAddLang(al,mkRow,gen);
 if(!al.querySelector('[tabindex="0"]')&&al.firstChild)al.firstChild.tabIndex=0;if(!sl.querySelector('[tabindex="0"]')&&sl.firstChild)sl.firstChild.tabIndex=0;
 if(fk)focusKey(fk)}
function buildAddLang(al,mkRow,gen){
 const am=$('audmore'),S=F.alData?F.alData():null;am.textContent='';am.hidden=!(S&&S.on);if(!S||!S.on)return;
 const have=[...document.querySelectorAll('#tracks .tt')].map(t=>t.dataset.trk);
 Object.keys(S.addT).filter(k=>!have.includes(k)).forEach(k=>{const a=S.addT[k];al.appendChild(mkRow('a:'+k,S.name(k),S.wait===k,F.alText(a),a.audio_state==='ready'?'voice':a.audio_state==='error'?'':'work',()=>{if(gen!==jobGen)return;if(a.audio_state==='error'){F.addLang(k);return}if(F.startWait(k))closePanel();else note('Could not switch to '+S.name(k)+'.')}))});
 const mk=(key,label,cls,act)=>{const b=document.createElement('button');b.type='button';b.className='ae '+(cls||'');b.dataset.key=key;b.tabIndex=-1;const nm=document.createElement('span');nm.className='nm';nm.textContent=label;b.appendChild(nm);b.addEventListener('click',act);am.appendChild(b);return b};
 mk('add','Add a language'+(S.open?' ▴':' ▾'),'add',()=>{F.alOpen(!S.open);buildPanel(true);focusKey('add')});
 if(S.open){const cand=S.cands;
  if(!cand.length){const p=document.createElement('p');p.className='mut';p.textContent='No more languages are available for this video.';am.appendChild(p)}
  cand.forEach(l=>mk('add:'+l.code,String(l.name||l.code),'addc',()=>{if(gen!==jobGen)return;F.alOpen(false);F.addLang(l.code);buildPanel(true);note('Adding '+String(l.name||l.code)+'. Playback continues.')}))}
 mk('replay','Replay sentence','',()=>{if(gen!==jobGen)return;if(F.replaySentence())closePanel();else note('No sentence to replay yet.')});
 if(!am.querySelector('[tabindex="0"]')&&!al.querySelector('[tabindex="0"]'))am.firstChild.tabIndex=0}

F.panelRefresh=()=>{if(panelOpen)buildPanel(true)};
function focusKey(k){const b=panel.querySelector('.ae[data-key="'+CSS.escape(k)+'"]');if(b)b.focus({preventScroll:true})}
const INERT=()=>[stagein,document.querySelector('.studio-grid'),hdr,$('landing'),$('foot')].filter(Boolean);
function openPanel(transient){lastFocus=document.activeElement;buildPanel(true);panel.classList.add('on');stage.classList.remove('panel','dim');stage.classList.add(transient?'dim':'panel');panelOpen=true;audbtn.classList.add('on');audbtn.setAttribute('aria-expanded','true');wake();clearTimeout(autoT);
 panel.setAttribute('role',transient?'status':'dialog');panel.setAttribute('aria-modal',transient?'false':'true');
 if(!transient){INERT().forEach(n=>{try{n.inert=true}catch(e){}});panel.classList.add('modal')}else{INERT().forEach(n=>{try{n.inert=false}catch(e){}});panel.classList.remove('modal')}
 const first=panel.querySelector('#audlist .ae[aria-checked=true]')||panel.querySelector('.ae');if(first&&!transient)first.focus({preventScroll:true});if(transient)scheduleClose(2600)}
function scheduleClose(ms){clearTimeout(autoT);autoT=setTimeout(()=>{if(!panel.matches(':hover')&&!panel.contains(document.activeElement))closePanel();else scheduleClose(1500)},ms)}
function closePanel(){clearTimeout(autoT);const had=panel.contains(document.activeElement);panel.classList.remove('on','modal');stage.classList.remove('panel','dim');panelOpen=false;audbtn.classList.remove('on');audbtn.setAttribute('aria-expanded','false');INERT().forEach(n=>{try{n.inert=false}catch(e){}});
 if(had){const to=(lastFocus&&lastFocus.isConnected&&lastFocus!==document.body)?lastFocus:audbtn;try{to.focus({preventScroll:true})}catch(e){}}lastFocus=null;wake()}
function togglePanel(){if(panelOpen)closePanel();else openPanel(false)}
document.addEventListener('keydown',e=>{
 if(e.key==='Escape'&&panelOpen){e.preventDefault();e.stopImmediatePropagation();closePanel();return}
 if(panelOpen&&panel.classList.contains('modal')&&!panel.contains(e.target)){e.preventDefault();e.stopImmediatePropagation();const f=panel.querySelector('.ae[tabindex="0"]')||panel.querySelector('.ae');if(f)f.focus({preventScroll:true});return}
 if(!panelOpen||!panel.contains(e.target))return;
 const items=[$('audclose'),...panel.querySelectorAll('.ae')],i=items.indexOf(e.target);
 if(e.key==='Tab'){const f=items[0],l=items[items.length-1];if(e.shiftKey&&e.target===f){e.preventDefault();l.focus()}else if(!e.shiftKey&&e.target===l){e.preventDefault();f.focus()}return}
 if(['ArrowDown','ArrowUp','ArrowLeft','ArrowRight','Home','End'].includes(e.key)){e.preventDefault();e.stopImmediatePropagation();if(i<0)return;
  const grp=[...e.target.parentNode.querySelectorAll('.ae')],gi=grp.indexOf(e.target);
  const n=e.key==='Home'?0:e.key==='End'?grp.length-1:(gi+(e.key==='ArrowDown'||e.key==='ArrowRight'?1:-1)+grp.length)%grp.length;grp[n].focus();return}
 if(e.key===' '||e.key==='Enter'){e.preventDefault();e.stopImmediatePropagation();if(i>=0)items[i].click();return}
 e.stopImmediatePropagation()},{capture:false});
$('pshield').addEventListener('click',e=>{e.stopPropagation();e.preventDefault();closePanel()});
stagein.addEventListener('click',e=>{if(panel.classList.contains('modal')){e.stopPropagation();e.preventDefault();closePanel()}},true);
stage.addEventListener('click',e=>{if(panelOpen&&!panel.contains(e.target)&&!audbtn.contains(e.target)&&!$('mstat').contains(e.target)){if(panel.classList.contains('modal'))e.stopPropagation();closePanel()}},true);
$('mstat').addEventListener('click',()=>togglePanel());
document.addEventListener('click',e=>{if(panelOpen&&panel.classList.contains('modal')&&!stage.contains(e.target)&&!e.composedPath().includes(panel))closePanel()});$('audclose').addEventListener('click',()=>closePanel());
$('tlang').addEventListener('change',()=>{if(panelOpen)buildPanel(false)});
$('capbtn').addEventListener('click',()=>{if(panelOpen)setTimeout(()=>buildPanel(false),0)});

/* language tabs: sliding underline, hide counters when complete, "Listen in" invitation, idle CTA + lockup */
function indicator(){const t=document.querySelector('#tracks .tt[aria-pressed=true]'),i=$('tind'),tr=$('tracks');if(!i)return;if(!t||!t.offsetParent){i.style.width='0';return}i.style.width=t.offsetWidth+'px';i.style.transform='translateX('+(t.offsetLeft-tr.scrollLeft)+'px)'}
addEventListener('resize',indicator);$('tracks').addEventListener('scroll',indicator,{passive:true});
if(document.fonts&&document.fonts.ready)document.fonts.ready.then(indicator).catch(()=>{});
let listenT=0,listenKey='';
function firstDub(){return [...document.querySelectorAll('#tracks .tt')].find(t=>t.dataset.trk!=='original'&&(t.querySelector('.dot.lips')||t.querySelector('.dot.voice')))}
function tracks(){
 document.querySelectorAll('#parts .pc').forEach(c=>{if(!c.title||c.title===c.dataset.t0){const t=(c.getAttribute('aria-label')||'').replace(/, \d+:\d\d to \d+:\d\d/,'');c.title=t;c.dataset.t0=t}});
 document.querySelectorAll('#tracks .tt').forEach(t=>{const n=t.querySelector('.n');const m=n&&/^(\d+)\/(\d+)$/.exec(n.textContent.trim());t.classList.toggle('full',!!(m&&m[1]===m[2]&&+m[2]>0))});
 indicator();if(panelOpen)buildPanel(false);
 const d=firstDub(),name=d?(d.textContent||'').replace(/\s*\d+\/\d+\s*$/,'').trim():'';
 const cta=$('ctaplay');if(d){$('ctatxt').textContent='Play in '+name;cta.hidden=false;cta.dataset.for=d.dataset.trk}else cta.hidden=true;
 const orig=document.querySelector('#tracks .tt[data-trk=original][aria-pressed=true]');
 if(orig&&d&&!v.paused){const key=(document.body.dataset.view)+':'+d.dataset.trk;if(listenKey!==key){listenKey=key;$('listentxt').textContent='Listen in '+name;$('listen').dataset.for=d.dataset.trk;$('listen').classList.add('on');clearTimeout(listenT);listenT=setTimeout(()=>$('listen').classList.remove('on'),6000)}}
 else{$('listen').classList.remove('on');if(!orig)listenKey=''}
 const sp=$('spdbtn');sp.classList.toggle('alt',(sp.textContent||'').trim()!=='1×')}
new MutationObserver(()=>{const sp=$('spdbtn');sp.classList.toggle('alt',(sp.textContent||'').trim()!=='1×')}).observe($('spdbtn'),{childList:true,subtree:true,characterData:true});
v.addEventListener('play',()=>setTimeout(tracks,50));
$('listen').addEventListener('click',()=>{const o=document.querySelector('#tracks .tt[data-trk="'+CSS.escape($('listen').dataset.for||'')+'"]');$('listen').classList.remove('on');if(o)o.click()});
let jobGen=0,playT=0;
const pressPlay=()=>{const g=jobGen,title=$('hdtitle').textContent;clearTimeout(playT);playT=setTimeout(()=>{if(g!==jobGen||document.body.dataset.view!=='studio'||$('hdtitle').textContent!==title)return;const bp=$('bigplay');if(bp&&!bp.disabled)bp.click()},60)};
$('ctaplay').addEventListener('click',()=>{const o=document.querySelector('#tracks .tt[data-trk="'+CSS.escape($('ctaplay').dataset.for||'')+'"]');if(o&&o.getAttribute('aria-pressed')!=='true')o.click();pressPlay()});
$('ctaorig').addEventListener('click',()=>{const o=document.querySelector('#tracks .tt[data-trk=original]');if(o&&o.getAttribute('aria-pressed')!=='true')o.click();pressPlay()});
/* ---- one-time keyboard hint, 1.5 s after the first play ---- */
let hintT=0;
function showHint(){let seen=false;try{seen=sessionStorage.getItem('dub-hint')==='1'}catch(e){}if(seen||hinted||innerWidth<768)return;hinted=true;try{sessionStorage.setItem('dub-hint','1')}catch(e){}const h=$('hint');clearTimeout(hintT);hintT=setTimeout(()=>{h.classList.add('on');hintT=setTimeout(()=>h.classList.remove('on'),3600)},1500)}
v.addEventListener('play',showHint);
/* ---- scrub ghost fill (playhead → cursor) ---- */
const bar=$('bar'),ghost=$('ghost');
bar.addEventListener('pointermove',e=>{const r=bar.getBoundingClientRect();if(!r.width)return;const f=Math.max(0,Math.min(1,(e.clientX-r.left)/r.width)),p=parseFloat($('fill').style.width)||0;const a=Math.min(f*100,p),b=Math.max(f*100,p);ghost.style.left=a+'%';ghost.style.width=(b-a)+'%'});
bar.addEventListener('pointerleave',()=>{ghost.style.width='0'});
/* ---- freeze-frame keyed by media generation (attach() owns freeze/unfreeze) + the idle poster, job-owned ---- */
const fz=$('freeze');let fzT=0,fzKey=null,postv=null,posterJob='',posterGen=0;
function paint(src){const r=stage.getBoundingClientRect();const dpr=Math.min(1.5,devicePixelRatio||1);const W=Math.round(r.width*dpr),H=Math.round(r.height*dpr);if(fz.width!==W)fz.width=W;if(fz.height!==H)fz.height=H;const g=fz.getContext('2d');if(!g)return;const vw=src.videoWidth,vh=src.videoHeight;const s=Math.min(W/vw,H/vh),w=vw*s,h=vh*s;g.clearRect(0,0,W,H);g.drawImage(src,(W-w)/2,(H-h)/2,w,h)}
function freeze(key){if(rm()||!fz)return;try{if(v.readyState<2||!v.videoWidth)return;paint(v);fz.classList.remove('poster');fz.classList.add('on');fzKey=key;clearTimeout(fzT);fzT=setTimeout(()=>unfreeze(key),2500)}catch(e){unfreeze(key)}}
function unfreeze(key){if(fzKey===null||(key!==undefined&&key!==fzKey))return;fzKey=null;clearTimeout(fzT);fz.classList.remove('on')}
v.addEventListener('play',()=>fz.classList.remove('poster'));
let postAC=null;
function poster(job){if(!job||!/^\w+$/.test(job)||job===posterJob)return;posterJob=job;const gen=++posterGen;
 if(!postv){postv=document.createElement('video');postv.id='postv';postv.muted=true;postv.preload='metadata';postv.playsInline=true;postv.setAttribute('aria-hidden','true');postv.style.cssText='position:absolute;width:1px;height:1px;opacity:0;pointer-events:none';document.body.appendChild(postv)}
 if(postAC)postAC.abort();const ac=new AbortController();postAC=ac;const src='/media/'+job+'/original/0.mp4';
 const live=()=>gen===posterGen&&!ac.signal.aborted&&postv.getAttribute('src')===src;
 postv.addEventListener('loadedmetadata',()=>{try{if(live())postv.currentTime=Math.min(8,Math.max(0,(postv.duration||10)*.2))}catch(e){}},{signal:ac.signal});
 postv.addEventListener('seeked',()=>{try{if(live()&&stage.classList.contains('idle')&&!v.getAttribute('src')&&postv.videoWidth){paint(postv);fz.classList.add('poster')}}catch(e){}posterUnload()},{signal:ac.signal});
 fz.classList.remove('poster');postv.src=src}
function posterUnload(){if(postAC){postAC.abort();postAC=null}if(postv&&postv.getAttribute('src')){try{postv.pause();postv.removeAttribute('src');postv.load()}catch(e){}}}
v.addEventListener('loadstart',posterUnload);
function posterStop(){posterGen++;posterJob='';if(postAC){postAC.abort();postAC=null}if(postv){try{postv.pause();postv.removeAttribute('src');postv.load()}catch(e){}}fz.classList.remove('poster','on');fzKey=null;clearTimeout(fzT);try{const g=fz.getContext('2d');if(g)g.clearRect(0,0,fz.width,fz.height)}catch(e){}}
/* ---- buffered ranges mapped into the part the player is on ---- */
let bufQ=0;
function buffered(){bufQ=0;const seg=document.querySelector('#segbar span.cur'),bd=$('bufd'),sb=$('segbar');if(!seg||!v.duration||!isFinite(v.duration)||!sb.offsetWidth){bd.textContent='';return}
 const W=sb.offsetWidth,L=seg.offsetLeft,SW=seg.offsetWidth,b=v.buffered;let h='';
 for(let i=0;i<b.length;i++){const s=Math.max(0,b.start(i)/v.duration),e=Math.min(1,b.end(i)/v.duration);if(e<=s)continue;h+='<i style="left:'+((L+s*SW)/W*100).toFixed(3)+'%;width:'+(((e-s)*SW)/W*100).toFixed(3)+'%"></i>'}
 bd.innerHTML=h}
const bufSched=()=>{if(!bufQ)bufQ=requestAnimationFrame(buffered)};
v.addEventListener('progress',bufSched);v.addEventListener('timeupdate',bufSched);v.addEventListener('emptied',bufSched);addEventListener('resize',bufSched);
/* ---- the island: ONE shell (#banner) with a priority queue; #toast (logic-written) and the switch pill become its states ---- */
const isl=$('banner'),ev=$('islev'),toast=$('toast'),lpill=$('lpill');
let islW=0,islH=0,islT=0,islSig='',islX=0,hover=false,sw=null,swT=0,phaseT=0,annT=0;
function curEvent(){
 if(toast.classList.contains('on')&&toast.classList.contains('e'))return{kind:'err',text:toast.textContent};
 if(sw)return{kind:'switch',name:sw.name,sub:sw.state};
 if(toast.classList.contains('on'))return{kind:'toast',text:toast.textContent};
 return null}
const evSig=e=>e?[e.kind,e.text||'',e.name||'',e.sub||''].join('|'):'';
function renderEvent(e){ev.textContent='';ev.classList.toggle('err',e.kind==='err');
 if(e.kind==='switch'){const i=document.createElement('i');i.textContent=e.name;ev.appendChild(i);if(e.sub){const s=document.createElement('s');s.textContent='· '+e.sub;ev.appendChild(s)}const c=document.createElement('span');c.innerHTML=ic('check');ev.appendChild(c.firstChild)}
 else{if(e.kind==='err'){const c=document.createElement('span');c.innerHTML=ic('alert');ev.appendChild(c.firstChild)}const t=document.createElement('span');t.textContent=e.text;ev.appendChild(t)}}
function renderPill(e){lpill.textContent='';if(!e){lpill.classList.remove('on');return}const i=document.createElement('i');i.textContent=e.name;lpill.appendChild(i);if(e.sub){const s=document.createElement('s');s.textContent='· '+e.sub;lpill.appendChild(s)}lpill.classList.add('on')}
function islandApply(){
 const e=curEvent();
 if(isl.hidden){islW=islH=0;islSig='';isl.__es='';isl.classList.remove('x','ev','gone');isl.style.width='';isl.style.height='';renderPill(e&&e.kind==='switch'&&!toast.classList.contains('on')?e:null);return}
 renderPill(null);
 [...$('bact').querySelectorAll('button')].forEach(b=>{if(/^dismiss$/i.test(b.textContent.trim()))b.hidden=true});
 const sig=(isl.dataset.k||'')+'|'+($('bt1').textContent||'')+'|'+($('bt2').textContent||'')+'|'+$('bact').children.length;
 if(sig!==islSig){islSig=sig;isl.__at=Date.now();isl.classList.remove('gone');clearTimeout(islX);islX=setTimeout(islSched,2500);clearTimeout(phaseT);phaseT=setTimeout(islSched,4600)}
 const k=isl.dataset.k,acts=$('bact').querySelector('button:not([hidden])'),age=Date.now()-(isl.__at||0);
 const persistent=k==='bad'||k==='warn'||!!acts;
 const wantX=!!$('bt2').textContent&&(hover||persistent||age<2400);
 const gone=!persistent&&!hover&&k==='ok'&&age>=4400&&!e;
 const wantEv=!!e,es=evSig(e);
 if(wantEv&&es!==isl.__es)renderEvent(e);isl.__es=es;
 if(wantX===isl.classList.contains('x')&&wantEv===isl.classList.contains('ev')&&gone===isl.classList.contains('gone')&&islW)return;
 isl.classList.toggle('gone',gone);
 const r0={w:islW,h:islH};
 isl.style.transition='none';isl.style.width='';isl.style.height='';isl.classList.toggle('x',wantX&&!wantEv);isl.classList.toggle('ev',wantEv);
 const r=isl.getBoundingClientRect();islW=Math.round(r.width);islH=Math.round(r.height);
 if(rm()||!r0.w||(r0.w===islW&&r0.h===islH)){void isl.offsetWidth;isl.style.transition='';return}
 isl.style.width=r0.w+'px';isl.style.height=r0.h+'px';void isl.offsetWidth;isl.style.transition='';isl.style.width=islW+'px';isl.style.height=islH+'px';
 clearTimeout(islT);islT=setTimeout(()=>{isl.style.transition='none';isl.style.width='';isl.style.height='';void isl.offsetWidth;isl.style.transition=''},520)}
let islQ=0;const islSched=()=>{if(islQ)return;islQ=requestAnimationFrame(()=>{islQ=0;islandApply()})};
new MutationObserver(islSched).observe(isl,{childList:true,subtree:true,characterData:true,attributes:true,attributeFilter:['data-k','hidden']});
new MutationObserver(islSched).observe(toast,{childList:true,characterData:true,subtree:true,attributes:true,attributeFilter:['class']});
isl.addEventListener('mouseenter',()=>{hover=true;islSched()});isl.addEventListener('mouseleave',()=>{hover=false;islSched()});
isl.addEventListener('focusin',()=>{hover=true;islSched()});isl.addEventListener('focusout',()=>{hover=false;islSched()});
function switched(name,state){if(/^original/i.test(name)&&/original/i.test(state||''))state='';sw={name,state};clearTimeout(swT);swT=setTimeout(()=>{sw=null;islSched()},2200);islSched();indicator();
 const ls=$('live-status');if(ls){ls.textContent='';clearTimeout(annT);const g=jobGen;annT=setTimeout(()=>{if(g===jobGen)ls.textContent='Audio: '+name+(state?', '+state:'')},40)}}
/* ---- reset: clears every presentation timer and transient state at job boundaries ---- */
function reset(){jobGen++;eatClear();[listenT,swT,autoT,phaseT,islX,islT,hintT,fzT,playT,annT].forEach(clearTimeout);sw=null;listenKey='';$('listen').classList.remove('on');$('hint').classList.remove('on');lpill.classList.remove('on');closePanel();panelSig='';islSig='';isl.__es='';isl.__at=0;isl.classList.remove('x','ev','gone');posterStop();bufOff();$('ctaplay').hidden=true;$('lktitle').textContent='';$('lkmeta').textContent='';stage.classList.remove('woke')}
/* ---- transcript display helpers (textContent only) ---- */
function sc(t){t=String(t==null?'':t);if(t.length>3&&t===t.toUpperCase()&&t!==t.toLowerCase()){t=t.toLowerCase();t=t.replace(/(^|[.!?]\s+)([a-z])/g,(m,a,b)=>a+b.toUpperCase());t=t.replace(/\bi\b/g,'I')}return t}
const HUES=[250,60,300,180,20,120,330,90,210,0];
function rows(){const seg=$('segs');let prev='';seg.querySelectorAll('.seg-row').forEach(r=>{const sw=r.querySelector('.sw'),sp=r.querySelector('.spk');const key=sp?sp.textContent.trim():'';r.classList.toggle('same',!!key&&key===prev);prev=key;
 if(sw){const m=/S(\d+)/.exec(sw.textContent||'');const n=m?(+m[1]-1):0;sw.style.background='oklch(72% .07 '+HUES[n%HUES.length]+')'}})}
/* ---- detail-page metadata (textContent only) ---- */
const KNOWN={'vTqc2wHT5A0':{t:'This Is Purdue',by:'Purdue University · Official University Podcast',src:'English',d:'Three speakers in conversation. Each voice is dubbed separately and the lips are matched to the new language.'},
 'R9IbfB0B4FU':{t:'Two Sides of the Same Star',by:'NASA ScienceCasts',src:'English',d:'Narration over footage with no visible speakers: the dub replaces the voice track and keeps the music and ambience.'},
 'vHJSRCkmXk4':{t:'Hindi to English',by:'YouTube',src:'Hindi',d:'A Hindi programme translated and voiced in English.'}};
function ytId(u){try{const x=new URL(u);const h=x.hostname.replace(/^www\./,'');if(h==='youtu.be')return x.pathname.slice(1).split('/')[0];if(/youtube/.test(h))return x.searchParams.get('v')||''}catch(e){}return''}
const fmt=s=>{s=Math.max(0,s|0);const m=s/60|0;return m+':'+String(s%60).padStart(2,'0')};
function meta(m){const T=$('vtitle'),C=$('vcredit'),D=$('vdesc'),M=$('vmore'),H=$('hdtitle'),P=$('psum');
 if(!m||!m.url&&!m.duration){T.textContent='Dubbing studio';C.textContent='';D.textContent='';M.textContent='';H.textContent='';P.textContent='voice first, then lip sync';F.src='';$('partspanel').classList.remove('done');return}
 const id=ytId(m.url||''),k=KNOWN[id]||{t:id?'YouTube video':'Your video',by:id?'youtube.com/watch?v='+id:'',src:'',d:''};F.src=k.src||'';
 T.textContent=k.t;D.textContent=k.d;H.textContent=k.t;$('lktitle').textContent=k.t;
 const mins=m.duration?Math.max(1,Math.round(m.duration/60))+' min':'';$('lkmeta').textContent=[mins,(m.langs||[]).join(', '),m.state==='done'?'Lips matched':''].filter(Boolean).join(' · ');
 C.textContent='';const add=(txt,cls)=>{const s=document.createElement('span');if(cls)s.className=cls;s.textContent=txt;C.appendChild(s)};
 if(k.by)add(k.by,'by');const bits=[];if(m.duration)bits.push(fmt(m.duration));if(m.parts)bits.push(m.parts+(m.parts===1?' part':' parts'));if(m.langs&&m.langs.length)bits.push(m.langs.length+(m.langs.length===1?' language':' languages'));if(bits.length)add(bits.join(' · '));
 if(m.langs&&m.langs.length)add('Audio: '+['Original'+(k.src?' ('+k.src+')':'')].concat(m.langs).join(', '));
 if(m.state!=='done')add(m.state==='error'?'Stopped':m.state==='queued'?'Queued':'Dubbing','tag');
 const allDone=m.state==='done'&&!!(m.langs&&m.langs.length)&&!document.querySelector('#parts .pc:not(.lips):not(.voice)');
 $('partspanel').classList.toggle('done',allDone);
 P.textContent=allDone?(m.parts+' parts · lips matched in '+m.langs.join(', ').replace(/, ([^,]*)$/,' and $1')):'voice first, then lip sync';
 M.textContent='';const row=(a,b)=>{const dt=document.createElement('dt');dt.textContent=a;const dd=document.createElement('dd');dd.textContent=b;M.appendChild(dt);M.appendChild(dd)};
 if(m.url)row('Source',String(m.url));if(m.job)row('Job',String(m.job));if(m.duration)row('Duration',fmt(m.duration));if(m.langs&&m.langs.length)row('Dubbed into',m.langs.join(', '));
 poster(m.job);if(panelOpen)buildPanel(false)}
/* ---- fullscreen: landscape lock on phones ---- */
document.addEventListener('fullscreenchange',()=>{try{if(document.fullscreenElement&&screen.orientation&&screen.orientation.lock&&innerWidth<900)screen.orientation.lock('landscape').catch(()=>{})}catch(e){}});
/* ---- theme crossfade ---- */
function themeFade(){if(rm())return;const f=$('tfade');f.style.background=getComputedStyle(document.body).backgroundColor;f.classList.remove('on');void f.offsetWidth;f.classList.add('on')}
/* initial icons + landing media */
$('muteic').innerHTML=ic('volume');$('mpp').innerHTML=ic('play');
if(document.body.dataset.view==='landing'){setTimeout(heroStart,300)}
return Object.assign(F,{ic,view,feedback,openPanel,closePanel,togglePanel,tracks,switched,meta,themeFade,buffered,freeze,unfreeze,reset,sc,rows});
})();
