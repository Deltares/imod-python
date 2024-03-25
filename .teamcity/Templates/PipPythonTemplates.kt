package Templates

import jetbrains.buildServer.configs.kotlin.DslContext
import jetbrains.buildServer.configs.kotlin.Template
import jetbrains.buildServer.configs.kotlin.buildSteps.script

object PipPython310Template : Template({
    name = "PipPython310Template"

    detectHangingBuilds = false

    vcs {
        root(DslContext.settingsRoot, "+:. => imod-python")

        cleanCheckout = true
    }

    steps {
        script {
            name = "Pip install python 3.10"
            id = "pip_install_py310"
            workingDir = "imod-python"
            scriptContent = """
                    pixi run --environment py310 --frozen test_import
                """.trimIndent()
            formatStderrAsError = true
        }
    }

    requirements {
        equals("env.OS", "Windows_NT")
    }
})

object PipPython311Template : Template({
    name = "PipPython311Template"

    detectHangingBuilds = false

    vcs {
        root(DslContext.settingsRoot, "+:. => imod-python")

        cleanCheckout = true
    }

    steps {
        script {
            name = "Pip install python 3.11"
            id = "pip_install_py311"
            workingDir = "imod-python"
            scriptContent = """
                    pixi run --environment py311 --frozen test_import
                """.trimIndent()
            formatStderrAsError = true
        }
    }

    requirements {
        equals("env.OS", "Windows_NT")
    }
})

object PipPython312Template : Template({
    name = "PipPython312Template"

    detectHangingBuilds = false

    vcs {
        root(DslContext.settingsRoot, "+:. => imod-python")

        cleanCheckout = true
    }

    steps {
        script {
            name = "Pip install python 3.12"
            id = "pip_install_py312"
            workingDir = "imod-python"
            scriptContent = """
                    pixi run --environment py312 --frozen test_import
                """.trimIndent()
            formatStderrAsError = true
        }
    }

    requirements {
        equals("env.OS", "Windows_NT")
    }
})