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

  // Browse… next to every folder box ([data-pick] = dialog title). The server shows the Windows folder dialog
  // (it runs on this PC) and returns the full path; [data-pick-append] adds a line instead of replacing.
  function addBrowseButtons() {
    document.querySelectorAll("[data-pick]:not([data-pick-ready])").forEach((field) => {
      field.dataset.pickReady = "1";
      const wrap = document.createElement("div");
      wrap.className = "pick-row";
      field.parentNode.insertBefore(wrap, field);
      wrap.appendChild(field);
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "btn";
      btn.textContent = "Browse…";
      btn.title = "Choose a folder";
      btn.addEventListener("click", async () => {
        const label = btn.textContent;
        btn.disabled = true;
        btn.textContent = "Choose in the dialog…";
        try {
          const lines = field.value.split("\n").map((s) => s.trim()).filter(Boolean);
          const body = new URLSearchParams({ initial: lines[lines.length - 1] || "", title: field.dataset.pick });
          const r = await fetch("/tools/pick-folder", { method: "POST", body });
          const { path } = await r.json();
          if (path) {
            if ("pickAppend" in field.dataset) {
              if (!lines.includes(path)) field.value = [...lines, path].join("\n");
            } else {
              field.value = path;
            }
            field.dispatchEvent(new Event("change", { bubbles: true }));
          }
        } catch (err) {
          alert("The folder dialog could not be opened. Paste the folder's path instead.");
        } finally {
          btn.disabled = false;
          btn.textContent = label;
        }
      });
      wrap.appendChild(btn);
    });
  }
  document.addEventListener("DOMContentLoaded", addBrowseButtons);
  document.addEventListener("htmx:afterSettle", addBrowseButtons);

  // Theme: System -> Light -> Dark. Remembered in this browser only.
  const THEMES = ["", "light", "dark"];
  function showTheme() {
    const btn = document.getElementById("theme");
    const t = document.documentElement.dataset.theme || "";
    if (btn) btn.textContent = "Theme: " + ({ "": "System", light: "Light", dark: "Dark" })[t];
  }
  document.addEventListener("click", (e) => {
    if (!e.target.closest("#theme")) return;
    const now = document.documentElement.dataset.theme || "";
    const next = THEMES[(THEMES.indexOf(now) + 1) % THEMES.length];
    if (next) document.documentElement.dataset.theme = next;
    else delete document.documentElement.dataset.theme;
    try { next ? localStorage.setItem("theme", next) : localStorage.removeItem("theme"); } catch (err) {}
    showTheme();
  });
  document.addEventListener("DOMContentLoaded", showTheme);
  document.addEventListener("htmx:afterSettle", showTheme);

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
