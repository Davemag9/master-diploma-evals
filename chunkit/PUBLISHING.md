# Publishing `md-chunkit` to npm

These steps are for the package author. Nothing has been published yet.

## 0. Check the name

```bash
npm view md-chunkit        # "404 Not Found" means the name is still free
```

On 2026-10-24, `md-chunkit`, `mdchunkit` and `chunkit-md` were free. `chunkit` itself is taken.
npm also rejects names that differ from an existing package only by punctuation, so `chunk-it` would be refused as too similar to `chunkit`.
If the name is taken by the time you publish, see the scoped option in section 2. The other choice is to rename the
package in `package.json` and `README.md`.

## 1. Create an npm account (one time)

1. Sign up at <https://www.npmjs.com/signup> and verify your email address.
2. Turn on two-factor authentication under **Account → Two-Factor Authentication**. npm requires 2FA, or a granular
   access token with 2FA bypass, before you can publish. Choose "Authorization and writes" and use an authenticator
   app or a security key.
3. Optional: under **Account → Profile**, add your full name ("Oleh Lozovyi") and GitHub.

## 2. Unscoped or scoped name

| | Unscoped: `md-chunkit` | Scoped: `@<your-username>/md-chunkit` |
|---|---|---|
| Install | `npm i md-chunkit` | `npm i @<your-username>/md-chunkit` |
| Name availability | must be free globally | always free inside your scope |
| Default visibility | public | **private** (a paid feature). Pass `--access public` to publish it free. |

To publish under a scope, set `"name": "@<your-username>/md-chunkit"` in `package.json`, update the install command and
imports in `README.md`, and always publish with `--access public`. You can also add
`"publishConfig": { "access": "public" }` to `package.json` so you don't forget.

## 3. Optional metadata

If the code is on GitHub, add these fields to `package.json`. They show up on the npm page:

```json
"repository": { "type": "git", "url": "git+https://github.com/<you>/md-chunkit.git" },
"homepage": "https://github.com/<you>/md-chunkit#readme",
"bugs": { "url": "https://github.com/<you>/md-chunkit/issues" }
```

## 4. Log in from the terminal

```bash
npm login                 # opens the browser (or prompts for user/password + 2FA code)
npm whoami                # should print your username
```

Check that you are logged in to the public registry: `npm config get registry` should print
`https://registry.npmjs.org/`.

## 5. Verify before publishing

```bash
cd /path/to/chunkit
npm ci                    # clean install of dev dependencies
npm run typecheck
npm test
npm run build
npm pack --dry-run        # inspect the file list: only dist/, README.md, LICENSE, package.json
```

Optional: install the packed tarball in a scratch project.

```bash
npm pack                  # creates md-chunkit-0.1.0.tgz
cd /tmp/try && npm init -y && npm i /path/to/chunkit/md-chunkit-0.1.0.tgz
node -e "console.log(require('md-chunkit').chunkIt('# A\n\nhello', 'qna'))"
```

`prepublishOnly` runs `typecheck`, `test` and `build` again automatically right before every `npm publish`.

## 6. Pick the version

The first release can go out as `0.1.0`, the version already in `package.json`. For later releases, bump the version
with `npm version`. If the folder is a git repository, this also creates a commit and a tag:

```bash
npm version patch         # 0.1.0 -> 0.1.1  bug fixes
npm version minor         # 0.1.1 -> 0.2.0  new backwards-compatible features
npm version major         # 0.2.0 -> 1.0.0  breaking API changes
```

## 7. Publish

```bash
npm publish --access public        # asks for a 2FA one-time code
# or: npm publish --access public --otp=123456
```

`--access public` is required for a scoped name and harmless for an unscoped one.

To test the release flow first, run `npm publish --dry-run`. To publish a pre-release without making it the default
`latest` version, run `npm version prerelease --preid beta && npm publish --tag beta --access public`.

## 8. Verify

```bash
npm view md-chunkit                 # metadata, versions, dist-tags
npm view md-chunkit version         # latest published version
npm view md-chunkit dist.tarball    # tarball URL
```

Also open <https://www.npmjs.com/package/md-chunkit> and check that the README renders. In a fresh folder, run
`npm i md-chunkit` and import it from both ESM (`import { chunkIt } from "md-chunkit"`) and CJS
(`const { chunkIt } = require("md-chunkit")`).

## 9. Later releases

1. Make your changes and update the tests and README.
2. Run `npm version patch|minor|major`.
3. Run `npm publish`. Add `--access public` for a scoped package unless `publishConfig` already sets it.
4. With git, run `git push --follow-tags`.

## Mistakes

- **Within 72 hours:** `npm unpublish md-chunkit@0.1.1` removes that version. A version number can never be reused,
  even after it is unpublished.
- **After that:** run `npm deprecate md-chunkit@0.1.1 "Broken, use 0.1.2"` and publish a fixed version.
