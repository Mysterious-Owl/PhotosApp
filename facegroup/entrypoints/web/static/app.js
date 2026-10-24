// Live status from the scheduler, and small selection helpers. No framework; htmx handles navigation.
(function () {
  let wasBusy = null;

  function connect() {
    const source = new EventSource("/events");
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

  document.addEventListener("click", (e) => {
    const btn = e.target.closest(".select-toggle");
    if (!btn) return;
    const form = document.getElementById(btn.dataset.target);
    const boxes = [...form.querySelectorAll('input[type=checkbox][name=face]')]
      .filter((b) => !btn.dataset.only || b.dataset[btn.dataset.only]);
    const turnOn = !boxes.every((b) => b.checked);
    boxes.forEach((b) => (b.checked = turnOn));
    if (btn.dataset.target === "review") btn.textContent = turnOn ? "Untick all" : "Tick all";
  });
})();
