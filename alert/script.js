const startBtn = document.getElementById("start-btn");
const status = document.getElementById("status");

let recognition;
const targetName = "Adil"; // <-- name to detect

if (!("webkitSpeechRecognition" in window)) {
  alert("Your browser does not support the Web Speech API");
} else {
  recognition = new webkitSpeechRecognition();
  recognition.continuous = true;
  recognition.interimResults = true;
  recognition.lang = "en-US";

  recognition.onresult = (event) => {
    let transcript = "";
    for (let i = event.resultIndex; i < event.results.length; i++) {
      transcript += event.results[i][0].transcript;
    }
    console.log("Transcript:", transcript);

    if (transcript.toLowerCase().includes(targetName.toLowerCase())) {
      document.body.style.backgroundColor = "red";
      status.textContent = `⚠ Name detected: ${targetName}`;
      if (navigator.vibrate) navigator.vibrate(500); // Vibrate for 500ms
    } else {
      document.body.style.backgroundColor = "#f0f0f0";
      status.textContent = "Status: Listening...";
    }
  };

  recognition.onerror = (event) => {
    console.error(event.error);
    status.textContent = "Error: " + event.error;
  };
}

startBtn.addEventListener("click", () => {
  recognition.start();
  status.textContent = "Status: Listening...";
});
