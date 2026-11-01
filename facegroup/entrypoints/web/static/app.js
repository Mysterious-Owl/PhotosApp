// Live status from the scheduler (server-sent events on /status/stream), and small selection helpers. No framework; htmx handles navigation.
(function () {
  let wasBusy = null;

  function connect() {
    const source = new EventSource("/status/stream");
    source.onmessage = (e) => {
      const s = JSON.parse(e.data);
      const el = document.getElementById("status");
      if (el) {
        el.className = "status status-" + s.state;
        el.querySelector(".msg").textContent = s.message;
      }
      const busy = !["idle", "error"].includes(s.state);
      // When indexing finishes, refresh the overview so new suggestions appear.
      if (wasBusy && !busy && location.pathname === "/people") {
        htmx.ajax("GET", "/people", { target: "body", swap: "outerHTML" });
      }
      wasBusy = busy;
    };
    source.onerror = () => { source.close(); setTimeout(connect, 3000); };
  }
  connect();

  function updateCount() {
    document.querySelectorAll(".selected-count").forEach((el) => {
      const form = el.closest("form");
      const n = form ? form.querySelectorAll("input[name=face]:checked").length : 0;
      el.textContent = n ? `${n} selected` : "";
    });
  }

  document.addEventListener("change", (e) => {
    if (e.target.matches("input[name=face]")) updateCount();
  });

  document.addEventListener("click", (e) => {
    const btn = e.target.closest(".select-toggle");
    if (!btn) return;
    const scope = document.getElementById(btn.dataset.target);
    const boxes = [...scope.querySelectorAll("input[type=checkbox][name=face]")]
      .filter((b) => !btn.dataset.only || b.dataset[btn.dataset.only]);
    const turnOn = !boxes.every((b) => b.checked);
    boxes.forEach((b) => (b.checked = turnOn));
    updateCount();
  });

  // Acting on nothing would be a confusing no-op: say so instead. Capture phase, so it runs before htmx.
  document.addEventListener("submit", (e) => {
    const form = e.target;
    if (!form.querySelector("input[name=face]") || form.querySelector("input[name=face]:checked")) return;
    if (e.submitter && e.submitter.name === "action") {
      e.preventDefault();
      e.stopImmediatePropagation();
      alert("Select one or more faces first.");
    }
  }, true);
})();
