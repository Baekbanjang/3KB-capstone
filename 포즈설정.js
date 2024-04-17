function toggleHelp() {
    console.log("toggleHelp 함수가 호출되었습니다.");
    var helpText = document.getElementById("helpText");
    if (helpText.style.display === "none" || helpText.style.display === "") {
        helpText.style.display = "block";
    } else {
        helpText.style.display = "none";
    }
}
