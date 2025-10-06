let lightMode = true;
const baseUrl = window.location.origin;

let pdfUploaded = false;       // Track if a file has been uploaded
let isProcessing = false;      // Prevent double-sending

// -------------------- LOADING ANIMATIONS --------------------
function showLoading() {
  $("#loading-dots").removeClass("d-none");
  $("#send-button").prop("disabled", true);
}

function hideLoading() {
  $("#loading-dots").addClass("d-none");
  $("#send-button").prop("disabled", false);
}

// -------------------- HELPERS --------------------
function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function cleanInput(value) {
  return value.trim().replace(/[\n\t]/g, "").replace(/<[^>]*>/g, "");
}

function scrollToBottom() {
  $("#chat-window").animate({ scrollTop: $("#chat-window")[0].scrollHeight });
}

function appendUserMessage(message) {
  $("#message-list").append(`
    <div class="message-line my-text">
      <div class="message-box my-text${!lightMode ? " dark" : ""}">
        <div class="me">${message}</div>
      </div>
    </div>
  `);
  scrollToBottom();
}

function appendBotMessage(message) {
  $("#message-list").append(`
    <div class="message-line">
      <div class="message-box${!lightMode ? " dark" : ""}">
        ${message}
      </div>
    </div>
  `);
  scrollToBottom();
}

// -------------------- MESSAGE PROCESSING --------------------
async function processUserMessage(message) {
  const response = await fetch(`${baseUrl}/process-message`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ userMessage: message }),
  });
  return await response.json();
}

// -------------------- DOCUMENT UPLOAD --------------------
$("#upload-form").on("submit", async function (e) {
  e.preventDefault();

  const fileInput = $("#pdf-upload")[0];
  const statusText = $("#upload-status");

  if (!fileInput.files.length) {
    statusText.text("Please select a PDF file before uploading.");
    return;
  }

  const file = fileInput.files[0];
  const formData = new FormData();
  formData.append("file", file);

  statusText.text("Uploading and processing your document...");
  showLoading();

  try {
    const response = await fetch(`${baseUrl}/process-document`, {
      method: "POST",
      body: formData,
    });

    const data = await response.json();
    appendBotMessage(data.botResponse);

    if (response.ok) {
      pdfUploaded = true;
      $("#upload-button").prop("disabled", true);
      $("#pdf-upload").prop("disabled", true);
      statusText.text("Document uploaded successfully! You can now ask questions.");
      $("#send-button").prop("disabled", false);
    } else {
      statusText.text("Failed to process the document. Please try again.");
    }
  } catch (err) {
    console.error("Upload error:", err);
    statusText.text("Something went wrong. Please try again.");
  } finally {
    hideLoading();
  }
});

// -------------------- CHAT HANDLING --------------------
async function handleSendMessage() {
  if (isProcessing) return;
  const message = cleanInput($("#message-input").val());
  if (!message) return;

  appendUserMessage(message);
  $("#message-input").val("");

  if (!pdfUploaded) {
    appendBotMessage("Please upload a PDF first before asking questions.");
    return;
  }

  showLoading();
  isProcessing = true;

  try {
    const response = await processUserMessage(message);
    appendBotMessage(response.botResponse);
  } catch (error) {
    console.error("Error sending message:", error);
    appendBotMessage("Sorry, something went wrong. Please try again.");
  } finally {
    hideLoading();
    isProcessing = false;
  }
}

// Send message on Enter key
$("#message-input").keyup(function (event) {
  if (event.keyCode === 13) {
    handleSendMessage();
  }
});

// Send message on button click
$("#send-button").click(async function () {
  await handleSendMessage();
});

// Reset chat
$("#reset-button").click(async function () {
  $("#message-list").empty();
  $("#upload-status").text("");
  $("#pdf-upload").val("");
  $("#upload-button").prop("disabled", false);
  $("#pdf-upload").prop("disabled", false);
  pdfUploaded = false;
  appendBotMessage("Welcome! Upload a PDF and I’ll summarize and answer your questions about it.");
});

// -------------------- LIGHT/DARK MODE --------------------
$("#light-dark-mode-switch").change(function () {
  $("body").toggleClass("dark-mode");
  $(".message-box").toggleClass("dark");
  $(".loading-dots").toggleClass("dark");
  $(".dot").toggleClass("dark-dot");
  lightMode = !lightMode;
});

// -------------------- INITIAL GREETING --------------------
$(document).ready(() => {
  appendBotMessage(
    "Hello! I’m <b>DocuMind Assistant</b>. Please upload a PDF document, and then ask me any questions — I’ll summarize and answer from it."
  );
});
