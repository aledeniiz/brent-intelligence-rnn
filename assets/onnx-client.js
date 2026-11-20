window.BrentONNX=(()=>{
 const root=new URL('./',document.currentScript.src);let session,manifest;
 const hex=buffer=>Array.from(new Uint8Array(buffer)).map(v=>v.toString(16).padStart(2,'0')).join('');
 async function load(){
  manifest=await fetch(new URL('model/manifest.json',root),{cache:'no-store'}).then(r=>{if(!r.ok)throw Error('Manifest no disponible');return r.json()});
  const bytes=await fetch(new URL('model/lstm.onnx',root)).then(r=>{if(!r.ok)throw Error('ONNX no disponible');return r.arrayBuffer()});
  if(hex(await crypto.subtle.digest('SHA-256',bytes))!==manifest.onnx_sha256)throw Error('Hash ONNX incompatible');
  ort.env.wasm.numThreads=1;ort.env.wasm.proxy=false;ort.env.wasm.wasmPaths=new URL('ort/',root).href;
  session=await ort.InferenceSession.create(bytes,{executionProviders:['wasm']});
 }
 async function infer(packet){
  if(!session)await load();
  if(packet.model_id!==manifest.model_id||packet.onnx_sha256!==manifest.onnx_sha256||packet.scaler_sha256!==manifest.scaler_sha256||JSON.stringify(packet.feature_names)!==JSON.stringify(manifest.feature_names))throw Error('Datos y modelo incompatibles');
  const origin=new Date(packet.origin.date+'T00:00:00Z'),age=(Date.now()-origin.getTime())/86400000;
  if(!Number.isFinite(age)||age<0||age>=8)throw Error('Entradas futuras u obsoletas: '+packet.origin.date);
  if(JSON.stringify(packet.input.shape)!=='[1,60,16]'||packet.input.values.length!==960||!packet.input.values.every(Number.isFinite))throw Error('Tensor inválido');
  if(!Number.isFinite(packet.origin.price)||packet.origin.price<=0)throw Error('Último Brent inválido');
  const inputArray=Float32Array.from(packet.input.values);
  if(packet.input.sha256&&hex(await crypto.subtle.digest('SHA-256',inputArray.buffer))!==packet.input.sha256)throw Error('Hash de entradas incompatible');
  const output=await session.run({features:new ort.Tensor('float32',Float32Array.from(packet.input.values),[1,60,16])});
  const normalized=Array.from(output.normalized_returns.data),scaler=manifest.scaler,q=manifest.calibration.log_error_quantiles;
  const horizons=[5,63,252].map((h,j)=>{const log=normalized[j]*scaler.y_std[j]+scaler.y_mean[j],price=packet.origin.price*Math.exp(log);return {sessions:h,price,delta_usd:price-packet.origin.price,delta_pct:100*(price/packet.origin.price-1),lower:price*Math.exp(-q[j]),upper:price*Math.exp(q[j]),nominal:.8,method:'Cuantil empírico de error logarítmico, calibración2022–2023 del mismo checkpoint. Sin garantía futura.',explanation:{5:'Escenario corto: ruido alto; comparar con el último precio.',63:'Escenario trimestral sensible a cambios económicos.',252:'Escenario anual con incertidumbre elevada; no trayectoria diaria.'}[h],limitations:['Pesos2021; inferencia ONNX, no reentrenamiento','Exógenas históricas sin certificación PIT','Sesiones observadas, no días naturales'],metrics:{...manifest.metrics[String(h)],improvement_pct:100*(1-manifest.metrics[String(h)].relative_rmse)}}});
  if(!horizons.every(h=>[h.price,h.lower,h.upper].every(Number.isFinite)))throw Error('Salida no finita');
  return {status:'ready',model:manifest.model_info,data:{last_date:packet.origin.date,last_price:packet.origin.price,downloaded_at:packet.generated_utc,sources:packet.used_sources,provider_sources:packet.sources},horizons,history:packet.history,events:packet.events||window.BRENT_EVENTS||[],warnings:['Inferencia ONNX real ejecutada en este navegador.','Calibración histórica80% nominal, no garantía futura.','Datos preparados por ingesta programada; actualizar no reentrena.'],uncertainty:{nominal:.8,calibration_n:251,calibration_start:'2022-01-03',calibration_end:'2022-12-29'},updated_at:new Date().toISOString(),inference:{engine:'onnxruntime-web-wasm',normalized}};
 }
 async function state(){const r=await fetch(new URL('data/latest.json',root),{cache:'no-store'});if(!r.ok)throw Error('Entradas actualizadas no disponibles');return infer(await r.json());}
 return {state,infer,get manifest(){return manifest}};
})();
