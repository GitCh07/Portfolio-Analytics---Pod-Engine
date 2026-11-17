// Render loop for the viewer. Loaded only when animations are switched on.
// One loop for the whole page, driving the active view. It stops itself when the
// view finishes, when the tab is hidden, or when the chart scrolls off-screen,
// and resumes when it is visible again.

export function createAnimator() {
  let view = null, el = null, raf = 0, last = 0, onScreen = true, wanted = false, io = null;

  const visible = () => onScreen && !document.hidden;

  function frame(ts) {
    raf = 0;
    if (!wanted || !view || !visible()) return;
    const dt = last ? Math.min((ts - last) / 1000, 0.1) : 0;
    last = ts;
    const more = view.tick(dt);
    if (more) raf = requestAnimationFrame(frame);
    else wanted = false;
  }
  function kick() {
    if (wanted && !raf && visible()) { last = 0; raf = requestAnimationFrame(frame); }
  }
  function onVis() { if (visible()) kick(); }

  document.addEventListener("visibilitychange", onVis);

  return {
    start(v, canvas) {
      this.stop();
      view = v; el = canvas; wanted = true; onScreen = true;
      v.playing = true;
      io = new IntersectionObserver((entries) => {
        onScreen = entries[entries.length - 1].isIntersecting;
        if (onScreen) kick();
      });
      io.observe(el);
      kick();
    },
    stop() {
      wanted = false;
      if (raf) cancelAnimationFrame(raf);
      raf = 0;
      if (io) { io.disconnect(); io = null; }
      if (view && view.playing) { view.playing = false; view.syncControls(); }
      view = null;
    },
    running: () => !!raf,
  };
}
