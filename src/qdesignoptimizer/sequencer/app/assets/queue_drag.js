// Drag and drop for the queue table.
//
// Plain HTML5 drag and drop, no library: Dash serves every file in
// app/assets/ on its own. Rows carry draggable="true" and data-step
// (layout.queue_table); a recipe's task rows carry data-parent, so a
// drop on them lands on the recipe.
//
// On drop this sends the new order of the written steps to the server
// through the "drag-order" store; callbacks.py writes it with
// Queue.reorder, which refuses an order the engine would not honour
// (a step above one it depends on, the running step moved). While a
// drag is under way "dragging" is true and the 2 s refresh pauses, so
// the table is not redrawn under the pointer.
(function () {
  "use strict";
  var dragged = null;
  var marked = null;

  function set(id, props) {
    var dc = window.dash_clientside;
    if (dc && dc.set_props) { dc.set_props(id, props); }
  }

  function rowOf(el) {
    var tr = el && el.closest ? el.closest("#queue-table tr") : null;
    if (!tr) { return null; }
    if (tr.dataset.step) { return tr; }
    if (tr.dataset.parent) {
      return document.querySelector(
        '#queue-table tr[data-step="' + CSS.escape(tr.dataset.parent) + '"]'
      );
    }
    return null;
  }

  function clearMark() {
    if (marked) {
      marked.classList.remove("qs-drop-before", "qs-drop-after");
      marked = null;
    }
  }

  function below(tr, ev) {
    var box = tr.getBoundingClientRect();
    return ev.clientY > box.top + box.height / 2;
  }

  document.addEventListener("dragstart", function (ev) {
    var tr = ev.target && ev.target.closest
      ? ev.target.closest("#queue-table tr[data-step]") : null;
    if (!tr || tr.getAttribute("draggable") !== "true") { return; }
    dragged = tr.dataset.step;
    tr.classList.add("qs-dragging");
    ev.dataTransfer.effectAllowed = "move";
    ev.dataTransfer.setData("text/plain", dragged);
    set("dragging", {data: true});
  });

  document.addEventListener("dragover", function (ev) {
    if (!dragged) { return; }
    var tr = rowOf(ev.target);
    if (!tr) { return; }
    ev.preventDefault();
    ev.dataTransfer.dropEffect = "move";
    var after = below(tr, ev);
    if (marked !== tr) { clearMark(); marked = tr; }
    tr.classList.toggle("qs-drop-after", after);
    tr.classList.toggle("qs-drop-before", !after);
  });

  document.addEventListener("drop", function (ev) {
    if (!dragged) { return; }
    var tr = rowOf(ev.target);
    if (!tr) { return; }
    ev.preventDefault();
    var target = tr.dataset.step;
    var after = below(tr, ev) || !!(ev.target.closest &&
      ev.target.closest("tr[data-parent]"));
    var order = Array.prototype.map.call(
      document.querySelectorAll("#queue-table tr[data-step]"),
      function (row) { return row.dataset.step; }
    ).filter(function (id) { return id !== dragged; });
    if (target !== dragged) {
      var at = order.indexOf(target) + (after ? 1 : 0);
      order.splice(at, 0, dragged);
      set("drag-order", {data: {order: order, at: Date.now()}});
    }
    clearMark();
  });

  document.addEventListener("dragend", function () {
    var tr = dragged && document.querySelector(
      '#queue-table tr[data-step="' + CSS.escape(dragged) + '"]');
    if (tr) { tr.classList.remove("qs-dragging"); }
    dragged = null;
    clearMark();
    set("dragging", {data: false});
  });
})();
