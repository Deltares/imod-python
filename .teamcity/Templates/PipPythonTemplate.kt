package Templates

import jetbrains.buildServer.configs.kotlin.DslContext
import jetbrains.buildServer.configs.kotlin.Template
import jetbrains.buildServer.configs.kotlin.buildFeatures.dockerSupport
import jetbrains.buildServer.configs.kotlin.buildSteps.ScriptBuildStep
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
            dockerImage = "containers.deltares.nl/hydrology_product_line_imod/windows-pixi:v0.26.1"
            dockerImagePlatform = ScriptBuildStep.ImagePlatform.Windows
            dockerRunParameters = """--cpus=4 --memory=16g"""
            dockerPull = true
        }
    }

    features {
        dockerSupport {
            loginToRegistry = on {
                dockerRegistryId = "PROJECT_EXT_134"
            }
        }
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
