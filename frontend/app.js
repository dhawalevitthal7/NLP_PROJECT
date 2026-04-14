const form = document.getElementById("evalForm");
const statusText = document.getElementById("statusText");
const jobIdText = document.getElementById("jobIdText");
const summary = document.getElementById("summary");
const resultBody = document.querySelector("#resultTable tbody");

let pollingTimer = null;

function setStatus(text) {
  statusText.textContent = text;
}

function clearTable() {
  resultBody.innerHTML = "";
}

function renderSummary(result) {
  const payload = {
    mode: result.mode,
    total_obtained: result.total_obtained,
    total_max_marks: result.total_max_marks,
    percentage: result.percentage,
    overall_confidence: result.overall_confidence,
    auto_graded_ratio: result.auto_graded_ratio,
    meta: result.meta,
  };
  summary.textContent = JSON.stringify(payload, null, 2);
}

function renderQuestions(questions) {
  clearTable();
  questions.forEach((q) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${q.sub_question_no ? `Q${q.question_no}(${q.sub_question_no})` : `Q${q.question_no}`}</td>
      <td>${q.obtained_marks}</td>
      <td>${q.max_marks}</td>
      <td>${q.confidence}</td>
      <td>${q.needs_manual_review ? "Yes" : "No"}</td>
      <td>${q.feedback}</td>
    `;
    resultBody.appendChild(tr);
  });
}

async function getJson(url) {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Request failed: ${res.status}`);
  }
  return res.json();
}

async function pollJob(jobId) {
  if (pollingTimer) {
    clearInterval(pollingTimer);
  }

  pollingTimer = setInterval(async () => {
    try {
      const status = await getJson(`/api/jobs/${jobId}/status`);
      setStatus(`${status.status} - ${status.progress_message || ""}`);

      if (status.status === "completed") {
        clearInterval(pollingTimer);
        pollingTimer = null;
        const result = await getJson(`/api/jobs/${jobId}/result`);
        renderSummary(result);
        renderQuestions(result.questions || []);
      }

      if (status.status === "failed") {
        clearInterval(pollingTimer);
        pollingTimer = null;
        summary.textContent = JSON.stringify(status, null, 2);
      }
    } catch (error) {
      clearInterval(pollingTimer);
      pollingTimer = null;
      setStatus(`Polling error: ${error.message}`);
    }
  }, 3000);
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  setStatus("Submitting...");
  summary.textContent = "Waiting for result...";
  clearTable();

  const markingFile = document.getElementById("markingPdf").files[0];
  const studentFile = document.getElementById("studentPdf").files[0];
  const mode = document.getElementById("mode").value;

  if (!markingFile || !studentFile) {
    setStatus("Please upload both PDF files.");
    return;
  }

  const formData = new FormData();
  formData.append("marking_scheme_pdf", markingFile);
  formData.append("student_answer_pdf", studentFile);

  try {
    const res = await fetch(`/api/evaluate?mode=${encodeURIComponent(mode)}`, {
      method: "POST",
      body: formData,
    });
    if (!res.ok) {
      throw new Error(`Upload failed: ${res.status}`);
    }
    const payload = await res.json();
    jobIdText.textContent = `Job ID: ${payload.job_id}`;
    setStatus("Processing started");
    pollJob(payload.job_id);
  } catch (error) {
    setStatus(`Error: ${error.message}`);
  }
});
