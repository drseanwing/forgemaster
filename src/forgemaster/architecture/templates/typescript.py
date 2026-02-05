"""TypeScript project template for repository scaffolding.

This module provides a comprehensive TypeScript project template with modern
tooling including Node.js, TypeScript, Jest/Vitest, ESLint, and Prettier.
"""

from __future__ import annotations

from forgemaster.architecture.scaffolder import ProjectTemplate


def get_typescript_template() -> ProjectTemplate:
    """Get TypeScript project template.

    Returns:
        ProjectTemplate configured for TypeScript projects

    Example:
        >>> template = get_typescript_template()
        >>> print(template.language)
        'typescript'
    """
    # Define directory structure
    directories = [
        "src",
        "tests",
        "dist",
    ]

    # Define file templates
    files = {
        "src/index.ts": _get_index_template(),
        "tests/index.test.ts": _get_test_index_template(),
        "README.md": _get_readme_template(),
        ".gitignore": _get_gitignore_template(),
    }

    # Define config files
    config_files = {
        "package.json": _get_package_json_template(),
        "tsconfig.json": _get_tsconfig_template(),
        ".eslintrc.json": _get_eslintrc_template(),
        ".prettierrc.json": _get_prettierrc_template(),
    }

    return ProjectTemplate(
        name="typescript",
        language="typescript",
        files=files,
        directories=directories,
        config_files=config_files,
    )


def _get_index_template() -> str:
    """Get index.ts template."""
    return '''/**
 * Main entry point for {{PROJECT_NAME}}
 */

/**
 * Main application function
 */
export async function main(): Promise<void> {
  console.log("Hello from {{PROJECT_NAME}}!");
}

// Run if executed directly (CommonJS)
// Note: This uses CommonJS module pattern as configured in tsconfig.json
if (require.main === module) {
  main().catch((error) => {
    console.error("Error:", error);
    process.exit(1);
  });
}
'''


def _get_test_index_template() -> str:
    """Get index.test.ts template."""
    return '''/**
 * Tests for main module
 */

import { main } from "../src/index";

describe("main", () => {
  it("should run without error", async () => {
    await expect(main()).resolves.not.toThrow();
  });
});
'''


def _get_readme_template() -> str:
    """Get README.md template."""
    return '''# {{PROJECT_NAME}}

{{PROJECT_DESCRIPTION}}

## Installation

```bash
# Install dependencies
npm install
```

## Usage

```bash
# Build the project
npm run build

# Run the application
npm start

# Run tests
npm test

# Type checking
npm run type-check

# Linting
npm run lint

# Format code
npm run format
```

## Development

This project uses:
- **TypeScript** for type-safe JavaScript
- **Jest** for testing
- **ESLint** for linting
- **Prettier** for code formatting

## Project Structure

```
src/
  index.ts
tests/
  index.test.ts
dist/
package.json
tsconfig.json
.eslintrc.json
.prettierrc.json
README.md
CLAUDE.md
```

## License

Copyright (c) 2026. All rights reserved.
'''


def _get_gitignore_template() -> str:
    """Get .gitignore template."""
    return '''# Dependencies
node_modules/
package-lock.json
yarn.lock
pnpm-lock.yaml

# Build output
dist/
build/
*.tsbuildinfo

# Testing
coverage/
.nyc_output/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# Logs
*.log
npm-debug.log*
yarn-debug.log*
yarn-error.log*

# OS
.DS_Store
Thumbs.db

# Environment
.env
.env.local
.env.*.local
'''


def _get_package_json_template() -> str:
    """Get package.json template."""
    return '''{
  "name": "{{PROJECT_NAME_LOWER}}",
  "version": "{{PROJECT_VERSION}}",
  "description": "{{PROJECT_DESCRIPTION}}",
  "main": "dist/index.js",
  "types": "dist/index.d.ts",
  "scripts": {
    "build": "tsc",
    "start": "node dist/index.js",
    "dev": "ts-node src/index.ts",
    "test": "jest",
    "test:watch": "jest --watch",
    "test:coverage": "jest --coverage",
    "type-check": "tsc --noEmit",
    "lint": "eslint src tests --ext .ts",
    "lint:fix": "eslint src tests --ext .ts --fix",
    "format": "prettier --write \\"src/**/*.ts\\" \\"tests/**/*.ts\\"",
    "format:check": "prettier --check \\"src/**/*.ts\\" \\"tests/**/*.ts\\""
  },
  "keywords": [],
  "author": "",
  "license": "UNLICENSED",
  "devDependencies": {
    "@types/jest": "^29.5.0",
    "@types/node": "^20.0.0",
    "@typescript-eslint/eslint-plugin": "^6.0.0",
    "@typescript-eslint/parser": "^6.0.0",
    "eslint": "^8.50.0",
    "jest": "^29.7.0",
    "prettier": "^3.0.0",
    "ts-jest": "^29.1.0",
    "ts-node": "^10.9.0",
    "typescript": "^5.3.0"
  },
  "dependencies": {},
  "engines": {
    "node": ">=20.0.0"
  }
}
'''


def _get_tsconfig_template() -> str:
    """Get tsconfig.json template."""
    return '''{
  "compilerOptions": {
    "target": "ES2022",
    "module": "commonjs",
    "lib": ["ES2022"],
    "outDir": "./dist",
    "rootDir": "./src",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true,
    "resolveJsonModule": true,
    "declaration": true,
    "declarationMap": true,
    "sourceMap": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noImplicitReturns": true,
    "noFallthroughCasesInSwitch": true,
    "moduleResolution": "node"
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules", "dist", "tests"]
}
'''


def _get_eslintrc_template() -> str:
    """Get .eslintrc.json template."""
    return '''{
  "parser": "@typescript-eslint/parser",
  "parserOptions": {
    "ecmaVersion": 2022,
    "sourceType": "module",
    "project": "./tsconfig.json"
  },
  "plugins": ["@typescript-eslint"],
  "extends": [
    "eslint:recommended",
    "plugin:@typescript-eslint/recommended",
    "plugin:@typescript-eslint/recommended-requiring-type-checking"
  ],
  "rules": {
    "@typescript-eslint/explicit-function-return-type": "warn",
    "@typescript-eslint/no-unused-vars": ["error", { "argsIgnorePattern": "^_" }],
    "@typescript-eslint/no-explicit-any": "warn",
    "no-console": "off"
  },
  "env": {
    "node": true,
    "es2022": true,
    "jest": true
  }
}
'''


def _get_prettierrc_template() -> str:
    """Get .prettierrc.json template."""
    return '''{
  "semi": true,
  "trailingComma": "es5",
  "singleQuote": false,
  "printWidth": 100,
  "tabWidth": 2,
  "useTabs": false,
  "arrowParens": "always",
  "endOfLine": "lf"
}
'''
