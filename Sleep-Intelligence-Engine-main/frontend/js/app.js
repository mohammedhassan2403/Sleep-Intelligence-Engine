document.getElementById("sleepForm").addEventListener("submit", async function (e) {
    e.preventDefault();

    const data = {
        sleep_duration: parseFloat(document.getElementById("sleep_duration").value),
        physical_activity_level: parseInt(document.getElementById("activity").value),
        stress_level: parseInt(document.getElementById("stress").value),
        heart_rate: parseInt(document.getElementById("heart_rate").value),
        daily_steps: parseInt(document.getElementById("steps").value)
    };

    const resultDiv = document.getElementById("result");
    resultDiv.innerHTML = "⏳ Analyzing sleep...";
    resultDiv.className = "loading";

    try {
        const response = await fetch("http://127.0.0.1:5000/api/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(data)
        });

        const result = await response.json();

        resultDiv.className = "success";
        resultDiv.innerHTML = `
            <h3>Sleep Score: ${result.sleep_score}</h3>
            <p><b>Pattern:</b> ${result.sleep_pattern}</p>
            <ul>
                ${result.recommendations.map(r => `<li>${r}</li>`).join("")}
            </ul>
        `;

    } catch (error) {
        resultDiv.className = "error";
        resultDiv.innerHTML = "❌ Backend not reachable. Start Flask server.";
    }
});
