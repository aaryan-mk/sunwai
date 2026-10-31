/* g08: cinematic layer. Presentation only: no DOM hook, id, data-* or API call of the app is touched. */
(function(){
'use strict';
var de=document.documentElement,body=document.body,$=function(i){return document.getElementById(i)};
var RMq=matchMedia('(prefers-reduced-motion: reduce)'),rm=function(){return RMq.matches};
var raf=window.requestAnimationFrame.bind(window);
/* ---------- preloader ---------- */
var pl=$('dubpl');
function plRun(){
 var t0=performance.now(),RM=rm(),MIN=RM?120:500,CAP=RM?600:1250,EX=RM?220:560,done=false,heroOk=RM,herov=$('herov'),T=[];
 try{sessionStorage.setItem('dub-pl','1')}catch(e){}
 if(RM)pl.classList.add('rm');
 de.dataset.pl='running';
 var heroEv=['loadeddata','playing','error'],GE=['pointerdown','mousedown','pointerup','mouseup','click','touchstart','keydown','keyup'];
 function hh(){heroOk=true;tryOut()}
 /* the dismissing gesture must not reach the page underneath: swallow it in the capture phase while the overlay is up */
 function gate(e){if(!pl.parentNode)return;if(done&&(e.type==='click'||e.type==='keyup')){e.preventDefault();e.stopPropagation();unbind();return}if((e.type==='keydown'||e.type==='keyup')&&e.key!=='Enter'&&e.key!==' ')return;e.preventDefault();e.stopPropagation();if(e.type==='pointerdown'||e.type==='keydown'||e.type==='touchstart')out()}
 function unbind(){GE.forEach(function(e){removeEventListener(e,gate,true)})}
 function cleanup(){T.forEach(clearTimeout);if(herov)heroEv.forEach(function(e){herov.removeEventListener(e,hh)});GE.forEach(function(e){removeEventListener(e,gate,true)})}
 function gone(){cleanup();de.classList.remove('pl-on','pl-hold');de.dataset.pl='done';if(pl.parentNode)pl.parentNode.removeChild(pl)}
 function out(){if(done)return;done=true;de.dataset.pl='leaving';pl.classList.add('out');de.classList.add('pl-go');
  if(herov)heroEv.forEach(function(e){herov.removeEventListener(e,hh)});
  T.push(setTimeout(function(){de.classList.remove('pl-hold')},RM?0:80));
  /* hard cap: whatever the exit animation is doing, the overlay is gone CAP ms after start */
  var rem=CAP-(performance.now()-t0),ex=Math.max(0,Math.min(EX,rem));T.push(setTimeout(function(){unbind();gone()},ex))}
 function tryOut(){if(done)return;if(performance.now()-t0>=MIN&&heroOk)out()}
 if(herov&&!RM){heroEv.forEach(function(e){herov.addEventListener(e,hh)});if(herov.readyState>=2)heroOk=true}
 else heroOk=true;
 T.push(setTimeout(tryOut,MIN));T.push(setTimeout(out,CAP-EX));
 if(document.hidden)out();
 GE.forEach(function(e){addEventListener(e,gate,{capture:true,passive:false})});
}
if(pl){if(de.classList.contains('pl-on'))plRun();else pl.parentNode.removeChild(pl)}

/* ---------- ambient layer over the hero ---------- */
var hero=document.querySelector('.hero'),amb=$('amb'),cv=$('ambc'),ctx=cv&&cv.getContext&&cv.getContext('2d');
var A={frames:0,running:false,n:0};window.__AMB=A;
if(hero&&amb&&ctx){
 var W=1,H=1,DPR=1,mx=0,my=0,tx=0,ty=0,sy=0,last=0,id=0,lowq=false,slow=0,heroVis=true,t=0;
 var BR=[[214,164,86],[232,150,60],[255,214,150],[200,120,50]];
 function sprite(c){var s=document.createElement('canvas');s.width=s.height=128;var g=s.getContext('2d'),r=g.createRadialGradient(64,64,0,64,64,64);r.addColorStop(0,'rgba('+c+',1)');r.addColorStop(.35,'rgba('+c+',.45)');r.addColorStop(1,'rgba('+c+',0)');g.fillStyle=r;g.fillRect(0,0,128,128);return s}
 var SP=BR.map(function(c){return sprite(c.join(','))});
 var GL=['अ','அ','অ','అ','ಅ','અ','ਅ','അ','ଅ','अ'];
 function gsprite(ch){var s=document.createElement('canvas');s.width=s.height=128;var g=s.getContext('2d');g.font='500 92px "Fraunces","Noto Sans Devanagari","Noto Sans Tamil","Noto Sans Bengali","Noto Sans Telugu","Noto Sans Kannada","Noto Sans Gujarati",sans-serif';g.textAlign='center';g.textBaseline='middle';g.fillStyle='rgb(236,192,120)';g.fillText(ch,64,70);return s}
 var GS=[];
 var rnd=function(a,b){return a+Math.random()*(b-a)};
 var orbs=[],glyphs=[],rib=[];
 for(var i=0;i<12;i++)orbs.push({x:rnd(0,1),y:rnd(0,1),r:rnd(50,170),z:rnd(.25,1),vx:rnd(-.006,.006),vy:rnd(-.01,-.002),a:rnd(.1,.3),c:i%4,p:rnd(0,6.28)});
 for(var j=0;j<9;j++)glyphs.push({x:rnd(.02,.98),y:rnd(.05,1),s:rnd(34,84),z:rnd(.3,1),vy:rnd(-.012,-.004),a:rnd(.05,.12),k:j,rot:rnd(-.3,.3),p:rnd(0,6.28)});
 for(var k=0;k<3;k++)rib.push({y:.62+k*.1,amp:rnd(14,34),f:rnd(.004,.009),sp:rnd(.3,.7)*(k%2?-1:1),a:.1-k*.02,z:.4+k*.25,ph:rnd(0,6)});
 function size(){var r=hero.getBoundingClientRect();W=Math.max(2,Math.round(r.width));H=Math.max(2,Math.round(r.height));DPR=Math.min(window.devicePixelRatio||1,1.25);cv.width=Math.round(W*DPR);cv.height=Math.round(H*DPR);ctx.setTransform(DPR,0,0,DPR,0,0)}
 function draw(dt){
  ctx.clearRect(0,0,W,H);
  mx+=(tx-mx)*Math.min(1,dt*3);my+=(ty-my)*Math.min(1,dt*3);
  var nO=lowq?6:(W<640?7:12),nG=lowq?4:(W<640?5:9),nR=lowq?1:3,sc=Math.min(window.scrollY,H*1.5)/H;
  ctx.globalCompositeOperation='lighter';
  for(var i=0;i<nO;i++){var o=orbs[i];o.x+=o.vx*dt;o.y+=o.vy*dt;if(o.y<-.3)o.y=1.3;if(o.x<-.2)o.x=1.2;if(o.x>1.2)o.x=-.2;
   var px=o.x*W+mx*o.z*34,py=o.y*H+my*o.z*22-sc*o.z*90,br=.75+.25*Math.sin(t*.5+o.p),r=o.r*(W<640?.7:1)*o.z+20;
   ctx.globalAlpha=o.a*br;ctx.drawImage(SP[o.c],px-r,py-r,r*2,r*2)}
  ctx.globalCompositeOperation='source-over';
  for(var g=0;g<nG;g++){var q=glyphs[g];if(!GS.length)break;q.y+=q.vy*dt;if(q.y<-.1)q.y=1.1;
   var gx=q.x*W+mx*q.z*46,gy=q.y*H+my*q.z*30-sc*q.z*140;
   ctx.save();ctx.translate(gx,gy);ctx.rotate(q.rot+Math.sin(t*.3+q.p)*.08);ctx.globalAlpha=q.a*(.7+.3*Math.sin(t*.6+q.p));ctx.drawImage(GS[q.k%GS.length],-q.s/2,-q.s/2,q.s,q.s);ctx.restore()}
  ctx.globalCompositeOperation='lighter';ctx.lineWidth=1.4;
  for(var m=0;m<nR;m++){var b=rib[m];ctx.globalAlpha=b.a;ctx.strokeStyle='rgb(222,172,96)';ctx.beginPath();
   var by=b.y*H+my*b.z*14-sc*b.z*60;
   for(var x=-10;x<=W+10;x+=22){var e=Math.sin(x/W*Math.PI),yy=by+Math.sin(x*b.f*6+t*b.sp+b.ph)*b.amp*e+Math.sin(x*b.f*17-t*b.sp*1.7)*b.amp*.28*e+mx*b.z*10;if(x<0)ctx.moveTo(x,yy);else ctx.lineTo(x,yy)}
   ctx.stroke()}
  ctx.globalCompositeOperation='source-over';ctx.globalAlpha=1}
 function frame(ts){id=raf(frame);if(!last)last=ts;var dt=Math.min(.05,(ts-last)/1000);last=ts;t+=dt;A.frames++;
  if(!lowq){if(dt>.026)slow++;else if(slow>0)slow--;if(slow>40)lowq=true}
  draw(dt)}
 function should(){return !rm()&&!document.hidden&&body.dataset.view==='landing'&&heroVis}
 function sync(){var go=should();
  if(go&&!id){last=0;id=raf(frame)}else if(!go&&id){cancelAnimationFrame(id);id=0}
  A.running=!!id;amb.dataset.run=id?'1':'0'}
 function boot(){size();GS=GL.map(gsprite);if(rm()){sync();amb.dataset.run='0';draw(0.016);return}sync()}
 A.sync=sync;
 try{var ro=new ResizeObserver(function(){size();if(!id)draw(.016)});ro.observe(hero)}catch(e){addEventListener('resize',size)}
 try{var io=new IntersectionObserver(function(es){heroVis=es[es.length-1].isIntersecting;sync()});io.observe(hero)}catch(e){}
 new MutationObserver(sync).observe(body,{attributes:true,attributeFilter:['data-view']});
 document.addEventListener('visibilitychange',sync);
 (RMq.addEventListener?RMq.addEventListener('change',function(){boot()}):RMq.addListener(boot));
 addEventListener('pointermove',function(e){if(e.pointerType==='touch')return;tx=(e.clientX/innerWidth-.5)*2;ty=(e.clientY/innerHeight-.5)*2},{passive:true});
 if(document.fonts&&document.fonts.ready)document.fonts.ready.then(function(){GS=GL.map(gsprite)});
 boot();
}

/* ---------- micro-interactions (mouse only, never in reduced motion) ---------- */
var fine=matchMedia('(hover:hover) and (pointer:fine)');
var demos=[].slice.call(document.querySelectorAll('.demo'));
demos.forEach(function(d){
 var pv=d.querySelector('.pv'),tm=0,pend=0,px=0,py=0;
 var ld=function(){d.classList.add('ld')};
 if(pv){pv.addEventListener('loadeddata',ld)}
 new MutationObserver(function(){if(d.querySelector('.pcap'))ld()}).observe(d,{childList:true});
 if(d.querySelector('.pcap'))ld();
 d.addEventListener('pointerenter',function(e){if(rm()||e.pointerType!=='mouse')return;clearTimeout(tm);tm=setTimeout(function(){d.classList.add('tl')},430)});
 d.addEventListener('pointermove',function(e){if(rm()||e.pointerType!=='mouse')return;px=e.clientX;py=e.clientY;if(pend)return;pend=raf(function(){pend=0;
  var r=d.getBoundingClientRect(),w=r.width/1.07,h=r.height/1.07,cx=r.left+r.width/2,cy=r.top+r.height/2,u=Math.max(-.5,Math.min(.5,(px-cx)/w)),v=Math.max(-.5,Math.min(.5,(py-cy)/h));
  d.style.setProperty('--ry',(u*13).toFixed(2)+'deg');d.style.setProperty('--rx',(-v*10).toFixed(2)+'deg');d.style.setProperty('--gx',((u+.5)*100).toFixed(1)+'%');d.style.setProperty('--gy',((v+.5)*100).toFixed(1)+'%')})});
 d.addEventListener('pointerleave',function(){clearTimeout(tm);d.classList.remove('tl');['--rx','--ry','--gx','--gy'].forEach(function(p){d.style.removeProperty(p)})});
});
/* magnetic primary buttons */
var mags=['heroplay','dub'].map($).filter(Boolean),mpend=0,mp=null;
addEventListener('pointermove',function(e){if(rm()||e.pointerType!=='mouse'||document.body.dataset.view!=='landing')return;mp=e;if(mpend)return;mpend=raf(function(){mpend=0;
 mags.forEach(function(b){var r=b.getBoundingClientRect(),cx=r.left+r.width/2,cy=r.top+r.height/2,dx=mp.clientX-cx,dy=mp.clientY-cy,d=Math.hypot(dx,dy),R=Math.max(r.width,r.height)/2+70;
  if(d<R){var k=.3;b.style.setProperty('--mx',Math.max(-9,Math.min(9,dx*k)).toFixed(1)+'px');b.style.setProperty('--my',Math.max(-7,Math.min(7,dy*k)).toFixed(1)+'px')}
  else if(b.style.getPropertyValue('--mx')){b.style.removeProperty('--mx');b.style.removeProperty('--my')}})})},{passive:true});
/* ripple / press */
document.addEventListener('pointerdown',function(e){if(rm()||e.button)return;var el=e.target.closest&&e.target.closest('.btn,#pp,.demo');if(!el||el.disabled)return;
 var cs=getComputedStyle(el);if(cs.position==='static')el.style.position='relative';
 var r=el.getBoundingClientRect(),s=document.createElement('span');s.className='rpl';s.setAttribute('aria-hidden','true');
 s.style.left=(e.clientX-r.left)+'px';s.style.top=(e.clientY-r.top)+'px';s.style.setProperty('--rs',Math.ceil(Math.max(r.width,r.height)*2.2/14));
 el.appendChild(s);setTimeout(function(){if(s.parentNode)s.parentNode.removeChild(s)},700)},true);
/* staggered scroll reveal of the rows */
if(!rm()&&'IntersectionObserver' in window){
 var items=[].slice.call(document.querySelectorAll('.rowh,.about>div')),io2=new IntersectionObserver(function(es){es.forEach(function(x){if(x.isIntersecting){x.target.classList.add('in');io2.unobserve(x.target)}})},{threshold:.15,rootMargin:'0px 0px -6% 0px'});
 items.forEach(function(el,i){if(el.getBoundingClientRect().top<innerHeight*.9)return;el.classList.add('rv');el.style.setProperty('--rd',(el.parentNode.classList.contains('about')?[].indexOf.call(el.parentNode.children,el)*110:0));io2.observe(el)});
 setTimeout(function(){document.querySelectorAll('.rv:not(.in)').forEach(function(el){if(el.getBoundingClientRect().top<innerHeight)el.classList.add('in')})},1500)}
/* language-switch sweep + island bloom */
var sw=$('swsweep');
document.addEventListener('click',function(e){if(rm()||!sw||body.dataset.view!=='studio')return;if(e.target.closest&&e.target.closest('.tt,.ae')){sw.classList.remove('go');void sw.offsetWidth;sw.classList.add('go')}},true);
function bloom(el){if(rm()||!el.animate||el.hidden)return;var base='0 10px 30px -10px rgba(0,0,0,.7),inset 0 0 0 1px rgba(255,255,255,.06)';
 try{el.animate([{boxShadow:base+',0 0 0 0 rgba(214,164,86,0)'},{boxShadow:base+',0 0 0 5px rgba(214,164,86,.38),0 0 34px rgba(214,164,86,.4)',offset:.3},{boxShadow:base+',0 0 0 12px rgba(214,164,86,0),0 0 0 rgba(214,164,86,0)'}],{duration:780,easing:'cubic-bezier(.16,1,.3,1)'})}catch(e){}}
['banner','toast','lpill'].forEach(function(n){var el=$(n);if(!el)return;var q=0;
 new MutationObserver(function(){if(q)return;q=setTimeout(function(){q=0;var nw=performance.now();if(nw-(el._bl||0)<1400)return;el._bl=nw;if(!el.hidden&&getComputedStyle(el).opacity!=='0')bloom(el)},30)}).observe(el,{attributes:true,attributeFilter:['class','data-k'],childList:n==='banner',subtree:n==='banner',characterData:n==='banner'})});
})();
