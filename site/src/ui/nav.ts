/*  The fixed bar's act indicator, and the chapter menu. */
import type { StoryScroll, StoryState } from "../motion/scroll";
import { $, $$, esc } from "./dom";

export class Nav {
  private num = $("#actNum")!;
  private name = $("#actName")!;
  private prog = $("#actProg")!;
  private menu = $("#menu")!;
  private btn = $<HTMLButtonElement>("#menuBtn")!;
  private lastAct = -1;
  private lastFocus: HTMLElement | null = null;

  constructor(private story: StoryScroll, private onNavigate: (href: string) => void) {
    $("#menuList")!.innerHTML = story.sections.map(s =>
      `<li><a href="#${s.el.id}" data-act="${s.act}"><span class="mono">${String(s.act).padStart(2, "0")}</span>${esc(s.name)}</a></li>`).join("");
    this.btn.addEventListener("click", () => this.toggle());
    addEventListener("keydown", e => { if (e.key === "Escape" && !this.menu.hidden) this.toggle(false); });
    this.menu.addEventListener("click", e => {
      const a = (e.target as HTMLElement).closest("a");
      if (!a) return;
      const href = a.getAttribute("href") || "";
      this.toggle(false);
      if (!href.startsWith("#/") && href.startsWith("#")) { e.preventDefault(); this.onNavigate(href); }
    });
    // keep focus inside the open menu
    this.menu.addEventListener("keydown", e => {
      if (e.key !== "Tab") return;
      const f = $$<HTMLElement>("a, button", this.menu).concat(this.btn);
      const i = f.indexOf(document.activeElement as HTMLElement);
      if (e.shiftKey && i <= 0) { e.preventDefault(); f[f.length - 1].focus(); }
      else if (!e.shiftKey && i === f.length - 1) { e.preventDefault(); f[0].focus(); }
    });
  }

  toggle(force?: boolean) {
    const open = force ?? this.menu.hidden !== false;
    this.menu.hidden = !open;
    this.btn.setAttribute("aria-expanded", String(open));
    this.btn.setAttribute("aria-label", open ? "Close chapter menu" : "Open chapter menu");
    document.documentElement.classList.toggle("menu-open", open);
    if (open) { this.lastFocus = document.activeElement as HTMLElement; this.story.lenis?.stop(); $<HTMLElement>("#menuList a")?.focus(); }
    else { this.story.lenis?.start(); this.lastFocus?.focus?.(); }
  }

  update(s: StoryState) {
    this.prog.parentElement!.style.setProperty("--p", s.actProgress.toFixed(3));
    if (s.act === this.lastAct) return;
    this.lastAct = s.act;
    document.documentElement.dataset.act = String(s.act);
    const sec = this.story.sections.find(x => x.act === s.act);
    this.num.textContent = String(s.act).padStart(2, "0");
    this.name.textContent = sec?.name || "";
    $$("#menuList a").forEach(a => {
      if (Number(a.dataset.act) === s.act) a.setAttribute("aria-current", "step");
      else a.removeAttribute("aria-current");
    });
  }
}
