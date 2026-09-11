const { spawnSync } = require('node:child_process');
const path = require('node:path');
const fs = require('node:fs');

// Usage: node scripts/build-dist.cjs <pyinstaller|nuitka>

const flavor = process.argv[2];
if (!flavor || !['pyinstaller', 'nuitka'].includes(flavor.toLowerCase())) {
  console.error('Usage: node scripts/build-dist.cjs <pyinstaller|nuitka>');
  process.exit(1);
}

const flavorCap = flavor.toLowerCase() === 'pyinstaller' ? 'PyInstaller' : 'Nuitka';

// Construct artifact name template for electron-builder
const artifactName = `\${productName}-${flavorCap}-\${version}-\${os}-\${arch}.\${ext}`;

console.log(`[build-dist] Flavor: ${flavorCap}`);
console.log(`[build-dist] Artifact Name Pattern: ${artifactName}`);

const projectRoot = path.resolve(__dirname, '..');

// Find electron-builder CLI script to run with node
// This avoids issues with symlinks, permissions, and shell diffs across platforms.
let ebCli = null;
try {
  // Try standard location for electron-builder v24+
  ebCli = require.resolve('electron-builder/out/cli/cli.js', { paths: [projectRoot] });
} catch (e) {
  try {
    // Try older location
    ebCli = require.resolve('electron-builder/cli.js', { paths: [projectRoot] });
  } catch (e2) {
    // Manual search fallback
    const possiblePaths = [
      path.join(projectRoot, 'node_modules', 'electron-builder', 'out', 'cli', 'cli.js'),
      path.join(projectRoot, 'node_modules', 'electron-builder', 'cli.js'),
    ];
    for (const p of possiblePaths) {
      if (fs.existsSync(p)) {
        ebCli = p;
        break;
      }
    }
  }
}

const args = [
  '--publish', 'never',
  `-c.artifactName=${artifactName}`
];

let r;

if (ebCli) {
  console.log(`[build-dist] Using electron-builder CLI: ${ebCli}`);
  // Run directly with node
  r = spawnSync(process.execPath, [ebCli, ...args], {
    cwd: projectRoot,
    stdio: 'inherit',
    shell: false
  });
} else {
  console.warn('[build-dist] Could not resolve electron-builder CLI script. Falling back to PATH lookup.');
  // Fallback to expecting 'electron-builder' in PATH (npm scripts usually have it)
  // Use shell: true to resolve command
  r = spawnSync('electron-builder', args, {
    cwd: projectRoot,
    stdio: 'inherit',
    shell: true
  });
}

if (r.error) {
  console.error('[build-dist] Spawn error:', r.error);
  process.exit(1);
}

if (r.status !== 0) {
  console.error(`[build-dist] Build failed with status ${r.status}`);
  process.exit(r.status === null ? 1 : r.status);
}
