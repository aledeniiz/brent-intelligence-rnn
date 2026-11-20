(()=>{
 const $=id=>document.getElementById(id),fmt=n=>new Intl.NumberFormat('es-ES',{minimumFractionDigits:2,maximumFractionDigits:2}).format(n);
 function sessions(from,to){let d=new Date(from+'T12:00:00Z'),end=new Date(to+'T12:00:00Z'),days=Math.round((end-d)/86400000);if(!Number.isFinite(days)||days<0)return 0;let n=Math.floor(days/7)*5;for(let i=0;i<days%7;i++){d.setUTCDate(d.getUTCDate()+1);if(d.getUTCDay()>0&&d.getUTCDay()<6)n++;}return n;}
 function estimate(){
  const state=window.UAXApp?.state,chosen=$('targetDate').value;
  if(!state||state.status!=='ready'){ $('dateResult').textContent='No hay un estado de modelo válido. No se calcula una estimación.';return;}
  if(!chosen){$('dateResult').textContent='Introduce una fecha objetivo válida.';return;}
  const origin=state.data.last_date;
  if(chosen<=origin){$('dateResult').textContent='La fecha debe ser posterior al último dato: '+origin+'. No se estiman precios históricos mediante este control.';return;}
  const n=sessions(origin,chosen),h=state.horizons.reduce((a,b)=>Math.abs(a.sessions-n)<=Math.abs(b.sessions-n)?a:b);
  const warning=n<5?'Fecha demasiado cercana: se usa la salida mínima de 5 sesiones.':n>252?'Fecha demasiado lejana: fuera del alcance de 252 sesiones; se muestra sólo la referencia más larga, sin extrapolar.':'Se usa el horizonte disponible más cercano; no existe una predicción específica para esa fecha.';
  $('dateResult').innerHTML='<div class="date-result-top"><span>Fecha elegida <strong>'+chosen+'</strong></span><span>Origen del escenario <strong>'+origin+'</strong></span></div><div class="date-stat"><strong>'+fmt(h.price)+' <small>USD/barril</small></strong><span>'+n+' sesiones estimadas → salida de '+h.sessions+' sesiones</span></div><p>Variación frente al precio de origen: <b>'+fmt(h.delta_usd)+' USD ('+fmt(h.delta_pct)+' %)</b>. Banda empírica '+Math.round(h.nominal*100)+' % nominal: <b>'+fmt(h.lower)+'–'+fmt(h.upper)+' USD</b>.</p><p class="date-warning">'+warning+'</p><p class="date-source">'+(state.synthetic?'ESCENARIO SINTÉTICO: salida ONNX real con entradas inventadas; NO es una previsión de mercado.':BrentAPI.mode==='demo'?'Instantánea archivada: no es una predicción actual ni una inferencia nueva.':'Respuesta del motor para un horizonte discreto; no es inferencia específica de fecha.')+'</p>';
  window.dateEstimate={chosen,origin,estimatedSessions:n,horizon:h.sessions,price:h.price,lower:h.lower,upper:h.upper,outOfRange:n<5||n>252};
  window.UAXApp.selectHorizon(h.sessions);
 }
 $('estimateDate').addEventListener('click',estimate);$('targetDate').addEventListener('change',estimate);
 window.BrentDate={sessions,estimate};
})();
