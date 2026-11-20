/* Static-first adapter: no automatic fallback from failed live API to old data. */
window.BrentAPI=(()=>{
 const cfg=window.BRENT_CONFIG||{mode:'demo'},live=cfg.mode==='api',onnx=cfg.mode==='onnx';let privateInput=null;
 const base=String(cfg.apiBase||'').replace(/\/$/,'');
 if(live&&!/^https?:\/\//.test(base))throw Error('Configura una URL API explícita antes de activar modo api.');
 async function state(){if(onnx){const value=await (privateInput?BrentONNX.infer(privateInput):BrentONNX.state());return new Response(JSON.stringify(value),{headers:{'Content-Type':'application/json'}});}return fetch(live?base+'/api/state':'./assets/demo-state.json',{cache:'no-store',headers:{Accept:'application/json'}})}
 async function refresh(signal){
  if(onnx)return state();
  if(!live)return new Response(JSON.stringify({error:'Modo demo: GitHub Pages no descarga datos ni ejecuta inferencia. Conecta una API para actualizar.'}),{status:409,headers:{'Content-Type':'application/json'}});
  return fetch(base+'/api/refresh',{method:'POST',headers:{'Content-Type':'application/json','X-UAX-Action':'refresh',Accept:'application/json'},body:'{}',signal});
 }
 window.addEventListener('DOMContentLoaded',()=>{
  const banner=document.querySelector('#connectionMode');
  banner.textContent=onnx?'Inferencia ONNX real en el navegador · pesos históricos identificados':live?'Modo API · valores recibidos del motor conectado':'Demo académica archivada · datos reales del 17/11/2026 · sin conexión al motor';
  const old=document.querySelector('#refreshButton');if(onnx&&old){old.textContent='Consultar datos publicados';old.title='Consulta las entradas generadas por ingesta; no dispara Actions ni descarga proveedores desde el navegador.';}else if(!live&&old){old.textContent='Actualizar requiere API';old.title='Esta publicación estática no ejecuta el backend';}
 });
 async function importPrivate(packet){await BrentONNX.infer(packet);privateInput=packet;return window.UAXApp.loadState();}
 return {state,refresh,importPrivate,mode:onnx?'onnx':live?'api':'demo'};
})();
