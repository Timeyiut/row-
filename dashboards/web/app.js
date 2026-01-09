const log = document.getElementById("log");
function set(id, v){ document.getElementById(id).textContent = v; }

const WS_URL = "ws://localhost:8765/ws";

function appendLog(line){
  log.textContent = (line + "\n" + log.textContent).slice(0, 6000);
}

let ws;
function connect(){
  ws = new WebSocket(WS_URL);
  ws.onopen = () => appendLog("WS connected: " + WS_URL);
  ws.onclose = () => { appendLog("WS disconnected. Reconnecting..."); setTimeout(connect, 1000); };
  ws.onerror = (e) => appendLog("WS error: " + e);
  ws.onmessage = (evt) => {
    try{
      const msg = JSON.parse(evt.data);
      if(msg.topic === "fusion/live"){
        if("hr" in msg.data){ set("hr", msg.data.hr); }
        if("hr_source" in msg.data){ set("hrsrc", msg.data.hr_source); }
        if("phase" in msg.data){ set("phase", msg.data.phase); }
        if(msg.data.angles){
          if("knee" in msg.data.angles) set("knee", msg.data.angles.knee);
          if("hip" in msg.data.angles) set("hip", msg.data.angles.hip);
          if("elbow" in msg.data.angles) set("elbow", msg.data.angles.elbow);
        }
      }
    }catch(err){
      appendLog("Bad JSON: " + evt.data);
    }
  };
}

appendLog("Connecting...");
connect();
