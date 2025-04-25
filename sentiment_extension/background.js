// background.js (Updated for Agent API)

chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "analyze-sentiment-agent", // Use a unique ID
    title: "Analyze Sentiment (Agent)", // Updated title
    contexts: ["selection"]
  });
});

chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId === "analyze-sentiment-agent" && info.selectionText) {
    // Indicate loading
    chrome.action.setBadgeText({ text: "..." });
    chrome.action.setBadgeBackgroundColor({ color: "#FFA500" }); // Orange

    const cleanedText = info.selectionText
      .replace(/[\u2018\u2019]/g, "'")
      .replace(/[\u201C\u201D]/g, '"')
      .replace(/\r?\n|\r/g, " ");

    const agentApiUrl = "http://127.0.0.1:5001/analyze_and_rewrite"; // Agent API URL

    fetch(agentApiUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: cleanedText })
    })
      .then(response => {
        // Try to parse JSON regardless of status code for error messages
        return response.json().then(data => ({ ok: response.ok, status: response.status, data }));
      })
      .then(({ ok, status, data }) => {
        if (!ok) {
          // Throw an error using the message from the API response if available
          throw new Error(data?.error || `Agent API error: ${status}`);
        }

        // Check if the expected structure is present (can be basic)
        if (data && data.analysis) {
          // Store the entire combined result
          chrome.storage.local.set({ agentAnalysisResult: data }, () => {
            if (chrome.runtime.lastError) {
              console.error("Error saving agent result to storage:", chrome.runtime.lastError);
              chrome.action.setBadgeText({ text: "ERR" });
              chrome.action.setBadgeBackgroundColor({ color: "#DC3545" });
            } else {
              chrome.action.setBadgeText({ text: "✓" });
              chrome.action.setBadgeBackgroundColor({ color: "#4688F1" });
              // Automatically open the popup
              chrome.action.openPopup().catch(err => console.error("Error opening popup:", err)); // Added catch
            }
          });
        } else {
          // Handle cases where response format is unexpected even with 2xx status
          throw new Error(data?.error || "Received invalid data structure from agent API.");
        }
      })
      .catch(err => {
        console.error("Agent API fetch error:", err);
        // Store the error message for the popup to display
        const errorObj = { error: err.message || err.toString() || "Unknown agent API error" };
        chrome.storage.local.set({ agentAnalysisResult: errorObj }, () => {
          // Update badge to show error
          chrome.action.setBadgeText({ text: "ERR" });
          chrome.action.setBadgeBackgroundColor({ color: "#DC3545" });
          // Optionally open popup to show error
          // chrome.action.openPopup().catch(err => console.error("Error opening popup:", err));
        });
      });
  }
});

// Listener to clear badge - can be triggered by popup or other events if needed
chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
    if (msg.action === "clearBadge") {
        chrome.action.setBadgeText({ text: "" });
        // Acknowledge message receipt (optional)
        // sendResponse({status: "Badge cleared"});
    }
    // Keep the message channel open for asynchronous responses if needed
    // return true;
});