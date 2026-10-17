import os
import sys
import sqlite3
import threading
import queue
import subprocess
import platform
from pathlib import Path
from datetime import datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog

from PIL import Image, ImageTk, ImageOps

APP_NAME = "Local Face Finder"
SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DB_DIR = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "LocalFaceFinder"
DB_PATH = DB_DIR / "facefinder.sqlite3"

# Matching threshold is intentionally conservative; adjust only after reviewing examples.
MATCH_THRESHOLD = 0.48


def open_path(path):
    try:
        if platform.system() == "Windows":
            os.startfile(str(path))
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except Exception as exc:
        messagebox.showerror("Could not open file", str(exc))


class Store:
    def __init__(self):
        DB_DIR.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.lock = threading.Lock()
        with self.conn:
            self.conn.execute("""CREATE TABLE IF NOT EXISTS photos (
                path TEXT PRIMARY KEY, mtime REAL NOT NULL, size INTEGER NOT NULL,
                scanned_at TEXT NOT NULL)""")
            self.conn.execute("""CREATE TABLE IF NOT EXISTS faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT, path TEXT NOT NULL,
                x1 INTEGER, y1 INTEGER, x2 INTEGER, y2 INTEGER,
                embedding TEXT NOT NULL, person_id INTEGER,
                FOREIGN KEY(path) REFERENCES photos(path))""")
            self.conn.execute("""CREATE TABLE IF NOT EXISTS people (
                id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL DEFAULT '')""")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_faces_path ON faces(path)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_faces_person ON faces(person_id)")

    def close(self):
        self.conn.close()

    def existing(self, path, stat):
        with self.lock:
            row = self.conn.execute("SELECT mtime,size FROM photos WHERE path=?", (str(path),)).fetchone()
        return bool(row and row["mtime"] == stat.st_mtime and row["size"] == stat.st_size)

    def save_photo(self, path, stat, faces):
        with self.lock, self.conn:
            self.conn.execute("DELETE FROM faces WHERE path=?", (str(path),))
            self.conn.execute("""INSERT OR REPLACE INTO photos(path,mtime,size,scanned_at)
                VALUES(?,?,?,?)""", (str(path), stat.st_mtime, stat.st_size, datetime.now().isoformat(timespec="seconds")))
            for face in faces:
                vec = ",".join(f"{float(v):.7f}" for v in face.embedding)
                self.conn.execute("""INSERT INTO faces(path,x1,y1,x2,y2,embedding,person_id)
                    VALUES(?,?,?,?,?,?,NULL)""",
                    (str(path), *[int(v) for v in face.bbox], vec))

    def all_faces(self):
        with self.lock:
            return self.conn.execute("""SELECT f.*, p.name AS person_name
                FROM faces f LEFT JOIN people p ON p.id=f.person_id ORDER BY f.path,f.id""").fetchall()

    def assign_face(self, face_id, person_id):
        with self.lock, self.conn:
            self.conn.execute("UPDATE faces SET person_id=? WHERE id=?", (person_id, face_id))

    def new_person(self, name=""):
        with self.lock, self.conn:
            cur = self.conn.execute("INSERT INTO people(name) VALUES(?)", (name,))
            return cur.lastrowid

    def rename_person(self, person_id, name):
        with self.lock, self.conn:
            self.conn.execute("UPDATE people SET name=? WHERE id=?", (name, person_id))

    def people(self):
        with self.lock:
            return self.conn.execute("""SELECT p.id,p.name,COUNT(f.id) AS count
                FROM people p LEFT JOIN faces f ON f.person_id=p.id
                GROUP BY p.id ORDER BY CASE WHEN p.name='' THEN 1 ELSE 0 END,p.name,p.id""").fetchall()

    def paths_for_person(self, person_id):
        with self.lock:
            return [r["path"] for r in self.conn.execute(
                "SELECT DISTINCT path FROM faces WHERE person_id=? ORDER BY path", (person_id,)).fetchall()]

    def photo_count(self):
        with self.lock:
            return self.conn.execute("SELECT COUNT(*) FROM photos").fetchone()[0]


def load_engine():
    # Import lazily so the interface can explain a missing dependency.
    from insightface.app import FaceAnalysis
    engine = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    engine.prepare(ctx_id=-1, det_size=(640, 640))
    return engine


def cosine(a, b):
    import numpy as np
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else -1.0


def group_unassigned(store, log):
    import numpy as np
    faces = store.all_faces()
    # Keep current assignments, and only cluster faces without a person_id.
    unassigned = [r for r in faces if r["person_id"] is None]
    if not unassigned:
        return
    vectors = []
    for r in unassigned:
        try:
            vectors.append(np.fromstring(r["embedding"], sep=",", dtype=np.float32))
        except Exception:
            vectors.append(np.zeros((512,), dtype=np.float32))
    # Greedy clustering: fast and conservative for a local-first MVP.
    clusters = []
    for row, vec in zip(unassigned, vectors):
        best_i, best_score = None, -1
        for i, cluster in enumerate(clusters):
            score = cosine(vec, cluster["centroid"])
            if score > best_score:
                best_i, best_score = i, score
        if best_i is not None and best_score >= MATCH_THRESHOLD:
            c = clusters[best_i]
            c["rows"].append(row)
            c["vectors"].append(vec)
            c["centroid"] = np.mean(c["vectors"], axis=0)
        else:
            clusters.append({"rows": [row], "vectors": [vec], "centroid": vec.copy()})
    created = 0
    for cluster in clusters:
        # Avoid cluttering the people list with single faces? Keep them too so every face is reviewable.
        pid = store.new_person("")
        for row in cluster["rows"]:
            store.assign_face(row["id"], pid)
        created += 1
    log(f"Grouped {len(unassigned)} faces into {created} provisional people. Review and rename groups.")


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_NAME)
        self.geometry("1120x760")
        self.minsize(900, 600)
        self.store = Store()
        self.events = queue.Queue()
        self.folder = tk.StringVar(value="")
        self.status = tk.StringVar(value="Choose a folder to begin. Photos remain unchanged.")
        self.selected_person = None
        self.current_paths = []
        self.thumb_refs = []
        self._build()
        self.after(150, self._poll_events)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build(self):
        self.configure(bg="#f4f6f8")
        header = tk.Frame(self, bg="#172033", padx=18, pady=14)
        header.pack(fill="x")
        tk.Label(header, text="Local Face Finder", bg="#172033", fg="white",
                 font=("Segoe UI", 20, "bold")).pack(side="left")
        tk.Label(header, text="PRIVATE • ON-DEVICE PHOTO ORGANIZER", bg="#172033", fg="#b9c5d8",
                 font=("Segoe UI", 9, "bold")).pack(side="right", pady=8)

        controls = ttk.Frame(self, padding=(14, 12))
        controls.pack(fill="x")
        ttk.Entry(controls, textvariable=self.folder).pack(side="left", fill="x", expand=True, padx=(0, 8))
        ttk.Button(controls, text="Choose Folder", command=self.choose_folder).pack(side="left", padx=4)
        self.scan_btn = ttk.Button(controls, text="Scan Folder", command=self.start_scan)
        self.scan_btn.pack(side="left", padx=4)
        ttk.Button(controls, text="Refresh Groups", command=self.refresh_people).pack(side="left", padx=4)

        main = ttk.Panedwindow(self, orient="horizontal")
        main.pack(fill="both", expand=True, padx=14, pady=(0, 10))
        left = ttk.Frame(main, padding=8)
        right = ttk.Frame(main, padding=8)
        main.add(left, weight=1)
        main.add(right, weight=3)

        ttk.Label(left, text="PEOPLE GROUPS", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0, 8))
        self.people_list = tk.Listbox(left, width=28, activestyle="dotbox", font=("Segoe UI", 10),
                                      borderwidth=0, highlightthickness=1, selectbackground="#dce8ff")
        self.people_list.pack(fill="both", expand=True)
        self.people_list.bind("<<ListboxSelect>>", self.on_person_select)
        person_actions = ttk.Frame(left)
        person_actions.pack(fill="x", pady=(8, 0))
        ttk.Button(person_actions, text="Name person", command=self.name_person).pack(fill="x", pady=2)
        ttk.Button(person_actions, text="Show all photos", command=self.show_all).pack(fill="x", pady=2)

        top = ttk.Frame(right)
        top.pack(fill="x")
        self.gallery_title = ttk.Label(top, text="Photo gallery", font=("Segoe UI", 13, "bold"))
        self.gallery_title.pack(side="left")
        ttk.Button(top, text="Open Original", command=self.open_selected).pack(side="right")
        ttk.Label(right, text="Select a thumbnail, then open the original file. Group matches are suggestions.",
                  foreground="#555").pack(anchor="w", pady=(4, 10))

        self.canvas = tk.Canvas(right, bg="white", highlightthickness=0)
        scrollbar = ttk.Scrollbar(right, orient="vertical", command=self.canvas.yview)
        self.gallery = ttk.Frame(self.canvas)
        self.gallery.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas_window = self.canvas.create_window((0, 0), window=self.gallery, anchor="nw")
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(self.canvas_window, width=e.width))
        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.canvas.bind_all("<MouseWheel>", lambda e: self.canvas.yview_scroll(int(-1*(e.delta/120)), "units"))

        footer = ttk.Frame(self, padding=(14, 8))
        footer.pack(fill="x")
        ttk.Label(footer, textvariable=self.status).pack(side="left", fill="x", expand=True)
        ttk.Label(footer, text=f"Index: {DB_PATH}", foreground="#666").pack(side="right")

    def log(self, msg):
        self.events.put(("status", msg))

    def choose_folder(self):
        folder = filedialog.askdirectory(title="Choose the folder containing your photos")
        if folder:
            self.folder.set(folder)

    def start_scan(self):
        folder = Path(self.folder.get().strip())
        if not folder.is_dir():
            messagebox.showwarning("Choose a folder", "Please choose a valid photo folder first.")
            return
        self.scan_btn.configure(state="disabled")
        self.status.set("Starting local face detector…")
        threading.Thread(target=self._scan_worker, args=(folder,), daemon=True).start()

    def _scan_worker(self, folder):
        try:
            self.log("Loading face model on this laptop. First launch may download model files.")
            engine = load_engine()
            paths = []
            for root, dirs, files in os.walk(folder):
                # Skip common hidden/system folders.
                dirs[:] = [d for d in dirs if not d.startswith(".") and d not in {"$RECYCLE.BIN", "System Volume Information"}]
                for filename in files:
                    p = Path(root) / filename
                    if p.suffix.lower() in SUPPORTED:
                        paths.append(p)
            total = len(paths)
            self.log(f"Found {total} supported image files. Scanning locally…")
            done = 0
            for path in paths:
                try:
                    stat = path.stat()
                    if self.store.existing(path, stat):
                        done += 1
                        if done % 25 == 0:
                            self.log(f"Scanned {done}/{total} files…")
                        continue
                    import cv2
                    image = cv2.imread(str(path))
                    if image is None:
                        self.store.save_photo(path, stat, [])
                    else:
                        faces = engine.get(image)
                        self.store.save_photo(path, stat, faces)
                except Exception as exc:
                    self.log(f"Skipped {path.name}: {exc}")
                done += 1
                if done % 10 == 0 or done == total:
                    self.log(f"Scanned {done}/{total} files…")
            group_unassigned(self.store, self.log)
            self.events.put(("scan_done", f"Finished. Indexed {self.store.photo_count()} photos."))
        except Exception as exc:
            self.events.put(("error", f"{type(exc).__name__}: {exc}\n\nCheck README.md for setup and model troubleshooting."))
        finally:
            self.events.put(("scan_enabled", None))

    def _poll_events(self):
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == "status":
                    self.status.set(value)
                elif kind == "scan_done":
                    self.status.set(value)
                    self.refresh_people()
                    self.show_all()
                elif kind == "scan_enabled":
                    self.scan_btn.configure(state="normal")
                elif kind == "error":
                    self.status.set("Scan stopped due to an error.")
                    messagebox.showerror("Scan error", value)
        except queue.Empty:
            pass
        self.after(150, self._poll_events)

    def refresh_people(self):
        people = self.store.people()
        self.people_list.delete(0, "end")
        self.people_rows = list(people)
        for row in people:
            name = row["name"].strip() or f"Unlabeled person {row['id']}"
            self.people_list.insert("end", f"{name}  ({row['count']} faces)")
        self.people_list.selection_clear(0, "end")
        self.selected_person = None

    def on_person_select(self, _event=None):
        sel = self.people_list.curselection()
        if not sel:
            return
        row = self.people_rows[sel[0]]
        self.selected_person = row["id"]
        paths = self.store.paths_for_person(self.selected_person)
        self.show_paths(paths, row["name"].strip() or "Unlabeled person")

    def name_person(self):
        if self.selected_person is None:
            messagebox.showinfo("Select a group", "Select a person group first.")
            return
        old = next((r["name"] for r in self.people_rows if r["id"] == self.selected_person), "")
        name = simpledialog.askstring("Name person", "Enter a label for this group:", initialvalue=old, parent=self)
        if name is not None:
            self.store.rename_person(self.selected_person, name.strip())
            self.refresh_people()
            self.status.set(f"Saved label: {name.strip() or '(unlabeled)'}")

    def show_all(self):
        self.selected_person = None
        self.gallery_title.configure(text="All indexed photos")
        with self.store.lock:
            rows = self.store.conn.execute("SELECT path FROM photos ORDER BY path").fetchall()
        self.show_paths([r["path"] for r in rows], "All indexed photos")

    def show_paths(self, paths, title):
        self.gallery_title.configure(text=title)
        self.current_paths = list(paths)
        self.thumb_refs = []
        for child in self.gallery.winfo_children():
            child.destroy()
        cols = 3
        for i, path in enumerate(self.current_paths):
            r, c = divmod(i, cols)
            card = ttk.Frame(self.gallery, padding=7, relief="ridge")
            card.grid(row=r, column=c, sticky="nsew", padx=5, pady=5)
            try:
                with Image.open(path) as im:
                    im = ImageOps.exif_transpose(im)
                    im.thumbnail((170, 130))
                    thumb = ImageTk.PhotoImage(im.copy())
                self.thumb_refs.append(thumb)
                img_label = tk.Label(card, image=thumb, bg="white", cursor="hand2")
                img_label.pack()
                img_label.bind("<Double-Button-1>", lambda e, p=path: open_path(p))
            except Exception:
                ttk.Label(card, text="[Preview unavailable]", width=22).pack(pady=30)
            name = Path(path).name
            ttk.Label(card, text=name[:28], width=24).pack(anchor="w", pady=(5, 0))
            ttk.Label(card, text=str(path)[:60], width=24, foreground="#666", wraplength=180).pack(anchor="w")
            ttk.Button(card, text="Open", command=lambda p=path: open_path(p)).pack(anchor="w", pady=(4, 0))
            card.bind("<Button-1>", lambda e, p=path: self.select_path(p))
        for c in range(cols):
            self.gallery.grid_columnconfigure(c, weight=1, uniform="gallery")

    def select_path(self, path):
        self.selected_path = path
        self.status.set(f"Selected: {path}")

    def open_selected(self):
        path = getattr(self, "selected_path", None)
        if path and Path(path).exists():
            open_path(path)
        else:
            messagebox.showinfo("Select a photo", "Click a photo card, then click Open Original.")

    def _on_close(self):
        try:
            self.store.close()
        finally:
            self.destroy()


if __name__ == "__main__":
    app = App()
    app.mainloop()
