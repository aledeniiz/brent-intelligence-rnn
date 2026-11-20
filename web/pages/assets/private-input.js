window.addEventListener('DOMContentLoaded',()=>{
 const section=document.createElement('section');section.className='date-explorer';section.innerHTML='<div><h2>Inferencia privada, en tu navegador.</h2><p>Este sitio público no distribuye las entradas financieras. Puedes cargar un JSON preparado con tus datos autorizados: el archivo permanece en esta pestaña y ONNX calcula las salidas reales.</p></div><div class="date-controls"><label for="privateInput">Archivo de entradas preparado (.json)</label><input type="file" id="privateInput" accept=".json,application/json"><p id="privateStatus" role="status" aria-live="polite">No se sube el archivo a ningún servidor. No se usan predicciones simuladas.</p></div>';
 document.querySelector('#dateExplorer').before(section);
 document.querySelector('#privateInput').addEventListener('change',async e=>{
  const file=e.target.files[0],status=document.querySelector('#privateStatus');if(!file)return;
  if(file.size>5*1024*1024){status.textContent='Archivo demasiado grande (máximo5MB).';return;}
  status.textContent='Validando identidad, fechas y tensor; ejecutando ONNX…';
  try{const state=await BrentAPI.importPrivate(JSON.parse(await file.text()));status.textContent=state?.status==='ready'?'Inferencia ONNX completada localmente. Ningún dato del archivo se ha publicado.':'No se ha obtenido un estado válido.';}
  catch(error){status.textContent='No se puede inferir: '+error.message;}
 });
});
