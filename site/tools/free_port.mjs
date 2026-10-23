/*  Print the first free TCP port at or after the one given.
 *
 *      node tools/free_port.mjs 4173 [host]
 *
 *  Used by start.bat / start.sh, so the links they print are the links that
 *  work (the server is then started with --strictPort on exactly this port).
 */
import { createServer } from "node:net";

const start = Number(process.argv[2] || 4173);
const host = process.argv[3] || "0.0.0.0";

const tryPort = port => new Promise(resolve => {
  const srv = createServer();
  srv.once("error", () => resolve(false));
  srv.once("listening", () => srv.close(() => resolve(true)));
  srv.listen(port, host);
});

for (let p = start; p < start + 50; p++) {
  if (await tryPort(p)) { console.log(p); process.exit(0); }
}
console.error(`no free port between ${start} and ${start + 49}`);
process.exit(1);
