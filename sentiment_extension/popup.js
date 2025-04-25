// popup.js (Updated for Agent API Results)
document.addEventListener("DOMContentLoaded", () => {
  const loadingMessage = document.getElementById("loading-message");
  const errorMessage = document.getElementById("error-message");
  const analysisContent = document.getElementById("analysis-content");

  // Get references to display areas
  const originalTextDisplay = document.getElementById("original-text-display");
  const sentimentDisplay = document.getElementById("sentiment-display");
  const emotionsDisplay = document.getElementById("emotions-display");
  const positiveRewriteDisplay = document.getElementById("positive-rewrite-display");
  const neutralRewriteDisplay = document.getElementById("neutral-rewrite-display");

  // Retrieve data stored by background.js
  chrome.storage.local.get(["agentAnalysisResult"], (result) => {
    loadingMessage.style.display = "none"; // Hide loading message

    if (chrome.runtime.lastError) {
      showError("Error retrieving analysis data: " + chrome.runtime.lastError.message);
      return;
    }

    const agentData = result.agentAnalysisResult;

    if (!agentData) {
      showError("No analysis data found. Please select text and right-click to analyze.");
      return;
    }

    // Check if the top-level object contains an error reported by background.js or the API
    if (agentData.error) {
      showError(`Analysis Error: ${agentData.error}`);
    }
    // Check if the expected analysis sub-object exists
    else if (agentData.analysis) {
      try {
        populateDisplay(agentData); // Populate using the new structure
        analysisContent.style.opacity = 1; // Fade in content
      } catch (e) {
        showError(`Error displaying results: ${e.message}`);
        console.error("Error building display:", e);
      }
    } else {
      showError("Analysis data structure is invalid.");
    }
  });

  function showError(message) {
     errorMessage.textContent = message;
     errorMessage.style.display = "block";
     analysisContent.style.display = "none";
  }

  // Function to populate the display areas using the new data structure
  function populateDisplay(data) {
    const analysis = data.analysis; // Analysis results are nested

    // Sanitize function
    const sanitize = (text) => text ? text.replace(/</g, "&lt;").replace(/>/g, "&gt;") : "N/A";

    // 1. Original Text
    originalTextDisplay.textContent = analysis.original_text || "N/A";

    // 2. Overall Sentiment
    sentimentDisplay.innerHTML = ''; // Clear
    if (analysis.sentiment && Array.isArray(analysis.sentiment) && analysis.sentiment.length > 0) {
      analysis.sentiment.forEach(sent => {
        const scorePct = Math.round(sent.score * 100);
        const color = sent.label === 'Positive' ? '#2ECC71' : sent.label === 'Negative' ? '#E74C3C' : '#95A5A6'; // Simplified color logic
        const sentimentItem = document.createElement('div');
        sentimentItem.className = 'sentiment-item';
        sentimentItem.innerHTML = `
          <span class="sentiment-label" style="color:${color};">${sent.label}</span>
          <div class="sentiment-track">
            <div class="sentiment-fill" style="width:${scorePct}%; background:${color};">
              ${scorePct}%
            </div>
          </div>
        `;
        sentimentDisplay.appendChild(sentimentItem);
      });
    } else {
      sentimentDisplay.innerHTML = '<p>Sentiment data unavailable.</p>';
    }

    // 3. Emotional Tone
    emotionsDisplay.innerHTML = ''; // Clear
    if (analysis.emotions && Array.isArray(analysis.emotions) && analysis.emotions.length > 0) {
      const validEmotions = analysis.emotions.filter(e => e.score > 0); // Ensure positive score
      if (validEmotions.length > 0) {
          const emotionColors = { // Define colors here or import
              "Fear":"#E74C3C", "Anger":"#C0392B", "Sadness":"#2980B9", "Disgust":"#8E44AD",
              "Surprise":"#F39C12", "Anticipation":"#D35400", "Trust":"#27AE60", "Joy":"#2ECC71",
              "Positive":"#2ECC71", "Negative":"#E74C3C", "Neutral":"#95A5A6"
          };
          validEmotions.forEach(e => {
            const pct = Math.round(e.score * 100);
            const color = emotionColors[e.emotion] || "#95A5A6"; // Get color or default
            const emotionItem = document.createElement('div');
            emotionItem.className = 'emotion-container';
            emotionItem.innerHTML = `
              <span class="emotion-label">${e.emotion}</span>
              <div class="emotion-track">
                <div class="emotion-fill" style="width:${pct}%; background:${color};">
                  ${pct > 5 ? pct + "%" : ""}
                </div>
              </div>
            `;
            emotionsDisplay.appendChild(emotionItem);
          });
       } else {
            emotionsDisplay.innerHTML = '<p>No specific emotions detected.</p>';
       }
    } else {
      emotionsDisplay.innerHTML = '<p>Emotion data unavailable.</p>';
    }

    // 4. Alternative Phrasings (Using AGENT results now)
    // Display results directly from the top level of the data object
    positiveRewriteDisplay.textContent = sanitize(data.agent_positive_rewrite || "Rewrite not generated.");
    neutralRewriteDisplay.textContent = sanitize(data.agent_neutral_rewrite || "Rewrite not generated.");
  }
});