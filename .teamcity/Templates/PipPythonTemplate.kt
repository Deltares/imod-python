package Templates

import jetbrains.buildServer.configs.kotlin.DslContext
import jetbrains.buildServer.configs.kotlin.Template
import jetbrains.buildServer.configs.kotlin.buildSteps.script
import jetbrains.buildServer.configs.kotlin.matrix

object PipPythonTemplate : Template({
    name = "PipPythonTemplate"

    detectHangingBuilds = false

    vcs {
        root(DslContext.settingsRoot, "+:. => imod-python")

        cleanCheckout = true
    }

    steps {
        script {
            name = "Pip install python"
            id = "pip_install"
            workingDir = "imod-python"
            scriptContent = """
                    pixi run --environment %python_env% --frozen test_import
                """.trimIndent()
            formatStderrAsError = true
        }
    }

    requirements {
        equals("env.OS", "Windows_NT")
    }

    features {
        matrix {
            param(
                "python_env", listOf(
                    value("py310", label = "python 3.10"),
                    value("py311", label = "python 3.11"),
                    value("py312", label = "python 3.12")
                )
            )
        }
    }
})
