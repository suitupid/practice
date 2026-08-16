function handleClearClick() {
    document.getElementById("input").value = "";
    document.getElementById("output").value = "";
}
const clearBtn = document.getElementById("clearBtn");
if(clearBtn){
    clearBtn.addEventListener("click", handleClearClick, false);
}

async function handleConvertClick() {
    const input = document.getElementById("input").value;
    console.log("Requested.");

    const response = await fetch("/api/convert", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({text: input})
    });
    const data = await response.json();

    console.log("Responsed.");
    document.getElementById("output").value = data.result;
}
const convertBtn = document.getElementById("convertBtn");
if(convertBtn){
    convertBtn.addEventListener("click", handleConvertClick, false);
}