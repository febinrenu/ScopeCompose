/*  Optional ambience, synthesised (no audio files). Off by default; only a
 *  click turns it on. It is never needed to understand anything: a low drone
 *  whose brightness follows scroll speed, and a soft tick at each new act.
 */
export class Ambience {
  private ctx: AudioContext | null = null;
  private master: GainNode | null = null;
  private filter: BiquadFilterNode | null = null;
  on = false;

  private build() {
    const C = (window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext);
    const ctx = new C();
    const master = ctx.createGain(); master.gain.value = 0;
    const filter = ctx.createBiquadFilter(); filter.type = "lowpass"; filter.frequency.value = 420; filter.Q.value = 0.7;
    filter.connect(master); master.connect(ctx.destination);
    for (const [f, g, d] of [[55, 0.5, 0], [82.4, 0.32, 3], [110.2, 0.18, -4]] as const) {
      const o = ctx.createOscillator(); o.type = "sine"; o.frequency.value = f; o.detune.value = d;
      const og = ctx.createGain(); og.gain.value = g;
      o.connect(og); og.connect(filter); o.start();
    }
    // breath: slow noise through a band-pass
    const len = ctx.sampleRate * 2, buf = ctx.createBuffer(1, len, ctx.sampleRate), data = buf.getChannelData(0);
    for (let i = 0; i < len; i++) data[i] = (Math.random() * 2 - 1) * 0.35;
    const n = ctx.createBufferSource(); n.buffer = buf; n.loop = true;
    const bp = ctx.createBiquadFilter(); bp.type = "bandpass"; bp.frequency.value = 900; bp.Q.value = 0.6;
    const ng = ctx.createGain(); ng.gain.value = 0.05;
    n.connect(bp); bp.connect(ng); ng.connect(filter); n.start();
    this.ctx = ctx; this.master = master; this.filter = filter;
  }

  async toggle(): Promise<boolean> {
    if (!this.ctx) this.build();
    const ctx = this.ctx!;
    if (ctx.state === "suspended") await ctx.resume();
    this.on = !this.on;
    this.master!.gain.setTargetAtTime(this.on ? 0.05 : 0, ctx.currentTime, 0.6);
    return this.on;
  }

  velocity(v: number) {
    if (!this.on || !this.ctx || !this.filter) return;
    const f = 380 + Math.min(1, Math.abs(v) / 40) * 1400;
    this.filter.frequency.setTargetAtTime(f, this.ctx.currentTime, 0.25);
  }

  tick(high = false) {
    if (!this.on || !this.ctx || !this.master) return;
    const ctx = this.ctx, o = ctx.createOscillator(), g = ctx.createGain();
    o.type = "sine"; o.frequency.value = high ? 1320 : 880;
    g.gain.setValueAtTime(0.0001, ctx.currentTime);
    g.gain.exponentialRampToValueAtTime(0.25, ctx.currentTime + 0.01);
    g.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + 0.35);
    o.connect(g); g.connect(this.master); o.start(); o.stop(ctx.currentTime + 0.4);
  }
}
