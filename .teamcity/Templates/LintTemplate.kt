package Templates

import jetbrains.buildServer.configs.kotlin.DslContext
import jetbrains.buildServer.configs.kotlin.Template
import jetbrains.buildServer.configs.kotlin.buildFeatures.dockerSupport
import jetbrains.buildServer.configs.kotlin.buildSteps.ScriptBuildStep
import jetbrains.buildServer.configs.kotlin.buildSteps.script

object LintTemplate : Template({
    name = "LintTemplate"

    detectHangingBuilds = false

    vcs {
        root(DslContext.settingsRoot, "+:. => imod-python")

        cleanCheckout = true
    }

    steps {
        script {
            name = "Static code analysis"
            id = "Static_code_analysis"
            workingDir = "imod-python"
            scriptContent = """
                    pixi run --environment default --frozen lint 
                """.trimIndent()
            formatStderrAsError = true
            dockerImage = "containers.deltares.nl/hydrology_product_line_imod/windows-pixi:v0.26.1"
            dockerImagePlatform = ScriptBuildStep.ImagePlatform.Windows
            dockerPull = true
        }
    }

    features {
        dockerSupport {
            loginToRegistry = on {
                dockerRegistryId = "PROJECT_EXT_134"
            }
        }
    }
})