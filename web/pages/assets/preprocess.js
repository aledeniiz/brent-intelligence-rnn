/* Exact frozen feature formulas for complete raw inputs; never fit scalers. */
window.BrentPreprocess={build(raw,m){
 const keys=['brent','wti','gas_natural','tipo_bce','usd_por_eur','cny_por_usd','vix','cobre','oro','indice_dolar'];
 if(!Array.isArray(raw)||raw.length<82)throw Error('Historial bruto insuficiente');
 raw.forEach((r,i)=>{if(!/^\d{4}-\d{2}-\d{2}$/.test(r.date)||!Number.isFinite(Date.parse(r.date))||new Date(r.date).toISOString().slice(0,10)!==r.date||(i&&r.date<=raw[i-1].date)||keys.some(k=>!Number.isFinite(r[k]))||r.brent<=0||r.vix<0||['gas_natural','usd_por_eur','cny_por_usd','cobre','oro','indice_dolar'].some(k=>r[k]<=0))throw Error('Datos brutos inválidos, incompletos o desordenados');});
 const lp=raw.map(r=>Math.log(r.brent)),ret=lp.map((v,i)=>i?v-lp[i-1]:NaN),avg=a=>a.reduce((s,v)=>s+v,0)/a.length;
 const values=[];
 for(let i=raw.length-60;i<raw.length;i++){
  const v=raw[i-1],previous=raw[i-6],r=ret.slice(i-20,i+1),mean=avg(r),log5=k=>Math.log(v[k])-Math.log(previous[k]);
  const f=[lp[i],ret[i],lp[i]-lp[i-5],lp[i]-lp[i-21],Math.sqrt(r.reduce((s,x)=>s+(x-mean)**2,0)/20),lp[i]-avg(lp.slice(i-20,i+1)),(v.wti-raw[i].brent)/raw[i].brent,(v.wti-previous.wti)/raw[i-5].brent,log5('gas_natural'),v.tipo_bce,log5('usd_por_eur'),log5('cny_por_usd'),Math.log1p(v.vix),log5('cobre'),log5('oro'),log5('indice_dolar')];
  f.forEach((x,j)=>values.push((x-m.scaler.x_mean[j])/m.scaler.x_std[j]));
 }
 if(values.length!==960||!values.every(Number.isFinite))throw Error('Características inválidas');
 return Array.from(Float32Array.from(values));
}};
