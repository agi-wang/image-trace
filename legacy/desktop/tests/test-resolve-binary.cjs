/**
 * Desktop startup and binary resolution tests.
 * Run with: node desktop/tests/test-resolve-binary.cjs
 */

const path = require('path');
const fs = require('fs');
const os = require('os');
const assert = require('assert');

let passed = 0;
let failed = 0;

function test(name, fn) {
    try {
        fn();
        console.log(`  ✅ ${name}`);
        passed++;
    } catch (e) {
        console.error(`  ❌ ${name}: ${e.message}`);
        failed++;
    }
}

// ─── Helpers ───
function mkTempDir() {
    return fs.mkdtempSync(path.join(os.tmpdir(), 'image-trace-test-'));
}

function cleanup(dir) {
    fs.rmSync(dir, { recursive: true, force: true });
}

function resolveBackendBinary(dir, platform = process.platform) {
    const suffix = platform === 'win32' ? '.exe' : '';
    const variants = [
        { flavor: 'nuitka', filename: `image-trace-backend-nuitka${suffix}`, subdir: 'image-trace-backend-nuitka' },
        { flavor: 'pyinstaller', filename: `image-trace-backend-pyinstaller${suffix}`, subdir: 'image-trace-backend-pyinstaller' },
        { flavor: 'nuitka-legacy', filename: `image-trace-backend-nuitka${suffix}`, subdir: null },
        { flavor: 'pyinstaller-legacy', filename: `image-trace-backend-pyinstaller${suffix}`, subdir: null },
        { flavor: 'legacy', filename: `image-trace-backend${suffix}`, subdir: null },
    ];

    const tried = [];
    for (const v of variants) {
        const full = v.subdir
            ? path.join(dir, v.subdir, v.filename)
            : path.join(dir, v.filename);
        tried.push(full);
        if (fs.existsSync(full)) {
            return { path: full, flavor: v.flavor.replace('-legacy', ''), tried, missing: false };
        }
    }

    return {
        path: path.join(dir, variants[0].subdir || '', variants[0].filename),
        flavor: variants[0].flavor,
        tried,
        missing: true,
    };
}

// ─── Test Suite: resolveBackendBinary ───
console.log('\n📦 resolveBackendBinary tests:');

test('finds nuitka in directory mode (macOS/Linux)', () => {
    const dir = mkTempDir();
    try {
        const subdir = path.join(dir, 'image-trace-backend-nuitka');
        fs.mkdirSync(subdir, { recursive: true });
        fs.writeFileSync(path.join(subdir, 'image-trace-backend-nuitka'), 'binary');
        const result = resolveBackendBinary(dir, 'darwin');
        assert.strictEqual(result.missing, false);
        assert.strictEqual(result.flavor, 'nuitka');
        assert.ok(result.path.includes('image-trace-backend-nuitka'));
    } finally {
        cleanup(dir);
    }
});

test('finds nuitka in directory mode (Windows)', () => {
    const dir = mkTempDir();
    try {
        const subdir = path.join(dir, 'image-trace-backend-nuitka');
        fs.mkdirSync(subdir, { recursive: true });
        fs.writeFileSync(path.join(subdir, 'image-trace-backend-nuitka.exe'), 'binary');
        const result = resolveBackendBinary(dir, 'win32');
        assert.strictEqual(result.missing, false);
        assert.strictEqual(result.flavor, 'nuitka');
        assert.ok(result.path.endsWith('.exe'));
    } finally {
        cleanup(dir);
    }
});

test('finds pyinstaller in directory mode', () => {
    const dir = mkTempDir();
    try {
        const subdir = path.join(dir, 'image-trace-backend-pyinstaller');
        fs.mkdirSync(subdir, { recursive: true });
        fs.writeFileSync(path.join(subdir, 'image-trace-backend-pyinstaller'), 'binary');
        const result = resolveBackendBinary(dir, 'linux');
        assert.strictEqual(result.missing, false);
        assert.strictEqual(result.flavor, 'pyinstaller');
    } finally {
        cleanup(dir);
    }
});

test('prefers nuitka over pyinstaller when both exist', () => {
    const dir = mkTempDir();
    try {
        // Create both
        const nDir = path.join(dir, 'image-trace-backend-nuitka');
        const pDir = path.join(dir, 'image-trace-backend-pyinstaller');
        fs.mkdirSync(nDir, { recursive: true });
        fs.mkdirSync(pDir, { recursive: true });
        fs.writeFileSync(path.join(nDir, 'image-trace-backend-nuitka'), 'binary');
        fs.writeFileSync(path.join(pDir, 'image-trace-backend-pyinstaller'), 'binary');
        const result = resolveBackendBinary(dir, 'darwin');
        assert.strictEqual(result.flavor, 'nuitka');
    } finally {
        cleanup(dir);
    }
});

test('falls back to legacy flat file', () => {
    const dir = mkTempDir();
    try {
        fs.writeFileSync(path.join(dir, 'image-trace-backend-pyinstaller'), 'binary');
        const result = resolveBackendBinary(dir, 'darwin');
        assert.strictEqual(result.missing, false);
        assert.strictEqual(result.flavor, 'pyinstaller');
    } finally {
        cleanup(dir);
    }
});

test('falls back to legacy name (image-trace-backend)', () => {
    const dir = mkTempDir();
    try {
        fs.writeFileSync(path.join(dir, 'image-trace-backend'), 'binary');
        const result = resolveBackendBinary(dir, 'darwin');
        assert.strictEqual(result.missing, false);
        assert.strictEqual(result.flavor, 'legacy');
    } finally {
        cleanup(dir);
    }
});

test('returns missing when nothing found', () => {
    const dir = mkTempDir();
    try {
        const result = resolveBackendBinary(dir, 'darwin');
        assert.strictEqual(result.missing, true);
        assert.ok(result.tried.length > 0, 'tried should list paths');
    } finally {
        cleanup(dir);
    }
});

test('tried contains all checked paths', () => {
    const dir = mkTempDir();
    try {
        const result = resolveBackendBinary(dir, 'darwin');
        assert.strictEqual(result.tried.length, 5, 'should try 5 variants');
    } finally {
        cleanup(dir);
    }
});

// ─── Test Suite: Build script validation ───
console.log('\n🔨 Build script validation tests:');

test('build-backend.cjs exists and does not use --onefile flag', () => {
    const script = path.resolve(__dirname, '..', 'scripts', 'build-backend.cjs');
    assert.ok(fs.existsSync(script), `File not found: ${script}`);
    const content = fs.readFileSync(script, 'utf-8');
    // Only check for --onefile as an actual argument (quoted), not in comments
    assert.ok(!content.match(/'--onefile'/), 'Should NOT have --onefile as argument');
    assert.ok(content.includes('--name'), 'Should have --name flag');
    assert.ok(content.includes('baseName'), 'Should define baseName');
});

test('build-backend-nuitka.cjs exists and uses --standalone without --onefile', () => {
    const script = path.resolve(__dirname, '..', 'scripts', 'build-backend-nuitka.cjs');
    assert.ok(fs.existsSync(script), `File not found: ${script}`);
    const content = fs.readFileSync(script, 'utf-8');
    // Only check for --onefile as an actual argument (quoted), not in comments
    assert.ok(!content.match(/["']--onefile["']/), 'Should NOT have --onefile as argument');
    assert.ok(content.includes('--standalone'), 'Should have --standalone');
    assert.ok(content.includes('--assume-yes-for-downloads'), 'Should auto-download deps on Windows');
    assert.ok(content.includes('run_server.dist'), 'Should handle Nuitka dist dir rename');
});

test('build-dist.cjs exists and handles flavor argument', () => {
    const script = path.resolve(__dirname, '..', 'scripts', 'build-dist.cjs');
    assert.ok(fs.existsSync(script), `File not found: ${script}`);
    const content = fs.readFileSync(script, 'utf-8');
    assert.ok(content.includes('pyinstaller'), 'Should handle pyinstaller flavor');
    assert.ok(content.includes('nuitka'), 'Should handle nuitka flavor');
});

// ─── Test Suite: package.json validation ───
console.log('\n📋 package.json validation tests:');

test('extraResources includes backend_bin', () => {
    const pkg = JSON.parse(fs.readFileSync(path.resolve(__dirname, '..', 'package.json'), 'utf-8'));
    const extra = pkg.build.extraResources;
    assert.ok(Array.isArray(extra), 'extraResources should be array');
    const hasBackendBin = extra.some(r => (typeof r === 'string' ? r : r.from) === 'backend_bin');
    assert.ok(hasBackendBin, 'extraResources should include backend_bin');
});

test('asarUnpack includes native modules', () => {
    const pkg = JSON.parse(fs.readFileSync(path.resolve(__dirname, '..', 'package.json'), 'utf-8'));
    const unpack = pkg.build.asarUnpack;
    assert.ok(Array.isArray(unpack), 'asarUnpack should be array');
    assert.ok(unpack.includes('**/*.dll'), 'Should unpack DLLs');
    assert.ok(unpack.includes('**/*.so'), 'Should unpack SOs');
    assert.ok(unpack.includes('**/*.dylib'), 'Should unpack dylibs');
});

test('no vc_redist in extraResources', () => {
    const pkg = JSON.parse(fs.readFileSync(path.resolve(__dirname, '..', 'package.json'), 'utf-8'));
    const extra = pkg.build.extraResources;
    const str = JSON.stringify(extra);
    assert.ok(!str.includes('vc_redist'), 'Should NOT include vc_redist (removed to prevent 0-byte upload)');
});

test('win target is nsis', () => {
    const pkg = JSON.parse(fs.readFileSync(path.resolve(__dirname, '..', 'package.json'), 'utf-8'));
    assert.deepStrictEqual(pkg.build.win.target, ['nsis']);
});

// ─── Test Suite: preload.js validation ───
console.log('\n🔌 preload.js validation tests:');

test('preload.js exposes required IPC methods', () => {
    const content = fs.readFileSync(path.resolve(__dirname, '..', 'preload.js'), 'utf-8');
    const required = ['startBackend', 'stopBackend', 'getBackendInfo', 'openDataDir', 'onBackendLog'];
    for (const method of required) {
        assert.ok(content.includes(method), `Should expose ${method}`);
    }
});

test('onBackendLog returns cleanup function', () => {
    const content = fs.readFileSync(path.resolve(__dirname, '..', 'preload.js'), 'utf-8');
    assert.ok(content.includes('removeListener'), 'Should return cleanup via removeListener');
});

// ─── Test Suite: main.js validation ───
console.log('\n⚡ main.js validation tests:');

test('main.js has sendLog function', () => {
    const content = fs.readFileSync(path.resolve(__dirname, '..', 'main.js'), 'utf-8');
    assert.ok(content.includes('function sendLog'), 'Should define sendLog');
    assert.ok(content.includes("backend:log"), 'Should send backend:log events');
});

test('main.js resolveBackendBinary checks subdirectories', () => {
    const content = fs.readFileSync(path.resolve(__dirname, '..', 'main.js'), 'utf-8');
    assert.ok(content.includes('subdir'), 'Should support subdir for directory mode');
    assert.ok(content.includes("'image-trace-backend-nuitka'"), 'Should check nuitka subdir');
    assert.ok(content.includes("'image-trace-backend-pyinstaller'"), 'Should check pyinstaller subdir');
});

test('main.js pipes stdout/stderr to renderer', () => {
    const content = fs.readFileSync(path.resolve(__dirname, '..', 'main.js'), 'utf-8');
    assert.ok(content.includes('[stdout]'), 'Should label stdout');
    assert.ok(content.includes('[stderr]'), 'Should label stderr');
});

// ─── CI workflow validation ───
console.log('\n🔄 CI workflow validation tests:');

test('desktop-build.yml has retention-days set to 1', () => {
    const content = fs.readFileSync(path.resolve(__dirname, '..', '..', '.github', 'workflows', 'desktop-build.yml'), 'utf-8');
    assert.ok(content.includes('retention-days: 1'), 'Retention should be 1 day');
});

test('desktop-build.yml has fail_on_unmatched_files: false', () => {
    const content = fs.readFileSync(path.resolve(__dirname, '..', '..', '.github', 'workflows', 'desktop-build.yml'), 'utf-8');
    assert.ok(content.includes('fail_on_unmatched_files: false'), 'Should not fail on missing files');
});

test('desktop-build.yml Release does not upload raw backend binaries', () => {
    const content = fs.readFileSync(path.resolve(__dirname, '..', '..', '.github', 'workflows', 'desktop-build.yml'), 'utf-8');
    // Only check the Release section (after softprops/action-gh-release)
    const releaseSection = content.split('softprops/action-gh-release')[1] || '';
    assert.ok(!releaseSection.includes('backend_bin_py'), 'Release should not upload backend_bin_py');
    assert.ok(!releaseSection.includes('backend_bin_nuitka'), 'Release should not upload backend_bin_nuitka');
});

test('desktop-build.yml uses Image Trace-*.exe glob (not **/*.exe)', () => {
    const content = fs.readFileSync(path.resolve(__dirname, '..', '..', '.github', 'workflows', 'desktop-build.yml'), 'utf-8');
    assert.ok(content.includes('Image Trace-*.exe'), 'Should use specific exe glob');
    assert.ok(!content.match(/\*\*\/\*\.exe/), 'Should NOT use **/*.exe glob');
});

// ─── Summary ───
console.log(`\n${'═'.repeat(50)}`);
console.log(`Results: ${passed} passed, ${failed} failed, ${passed + failed} total`);
console.log(`${'═'.repeat(50)}\n`);

process.exit(failed > 0 ? 1 : 0);
