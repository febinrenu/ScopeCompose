/*  Print the addresses a phone on the same network can open.
 *
 *      node tools/lan_links.mjs 4173
 *
 *  Used by start.bat. Private home/office ranges (192.168.x, 10.x, 172.16-31.x)
 *  are listed first; link-local and loopback addresses are skipped.
 */
import { networkInterfaces } from "node:os";

const port = process.argv[2] || "4173";
const isPrivate = a => /^(192\.168\.|10\.|172\.(1[6-9]|2\d|3[01])\.)/.test(a);
const found = [];
for (const [name, list] of Object.entries(networkInterfaces())) {
  for (const i of list || []) {
    if (i.family !== "IPv4" || i.internal || i.address.startsWith("169.254.")) continue;
    // virtual adapters (WSL, Hyper-V, VirtualBox, VMware) are not reachable from a phone
    if (/vEthernet|WSL|Hyper-V|VirtualBox|VMware|Loopback/i.test(name)) continue;
    found.push({ address: i.address, name });
  }
}
found.sort((a, b) => Number(isPrivate(b.address)) - Number(isPrivate(a.address)));
if (!found.length) console.log("     (no network connection found - connect to Wi-Fi and run start.bat again)");
for (const { address, name } of found) console.log(`     http://${address}:${port}      (${name})`);
