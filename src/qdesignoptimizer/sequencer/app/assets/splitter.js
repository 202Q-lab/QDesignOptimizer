// The divider between the left side (shared queue, project) and the
// study editor: drag it to share the width. The width is kept in this
// browser (localStorage), so it stays across reloads.
(function () {
  "use strict";
  var KEY = "qdo-sequencer-left";
  var root = document.documentElement;

  try {
    var kept = window.localStorage.getItem(KEY);
    if (kept) { root.style.setProperty("--qs-left", kept); }
  } catch (e) { /* no storage: the default width */ }

  document.addEventListener("mousedown", function (ev) {
    var bar = ev.target.closest ? ev.target.closest("#qs-splitter") : null;
    if (!bar) { return; }
    var main = document.getElementById("qs-main");
    if (!main) { return; }
    ev.preventDefault();
    bar.classList.add("qs-active");
    var box = main.getBoundingClientRect();

    function move(e) {
      var px = Math.min(Math.max(e.clientX - box.left, 280), box.width - 320);
      root.style.setProperty("--qs-left", px + "px");
    }
    function up() {
      bar.classList.remove("qs-active");
      document.removeEventListener("mousemove", move);
      document.removeEventListener("mouseup", up);
      try {
        window.localStorage.setItem(KEY, root.style.getPropertyValue("--qs-left"));
      } catch (e) { /* not kept */ }
    }
    document.addEventListener("mousemove", move);
    document.addEventListener("mouseup", up);
  });
}());
