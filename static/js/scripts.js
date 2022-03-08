function toggleLoadingGif() {
    var x = document.getElementById("loading-gif");
    if (x.style.display === "none") {
      x.style.display = "block";
    } else {
      x.style.display = "none";
    }
}