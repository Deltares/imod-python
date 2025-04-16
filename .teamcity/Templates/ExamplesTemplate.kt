package Templates

import jetbrains.buildServer.configs.kotlin.AbsoluteId
import jetbrains.buildServer.configs.kotlin.DslContext
import jetbrains.buildServer.configs.kotlin.Template
import jetbrains.buildServer.configs.kotlin.buildFeatures.XmlReport
import jetbrains.buildServer.configs.kotlin.buildFeatures.dockerSupport
import jetbrains.buildServer.configs.kotlin.buildFeatures.xmlReport
import jetbrains.buildServer.configs.kotlin.buildSteps.ScriptBuildStep
import jetbrains.buildServer.configs.kotlin.buildSteps.script

object ExamplesTemplate : Template({
    name = "ExamplesTemplate"

    artifactRules = """imod-python\imod\tests\temp => test_output.zip"""

    vcs {
        root(DslContext.settingsRoot, "+:. => imod-python")
        root(AbsoluteId("iMOD6_IMODPython_MetaSwapLookupTable"), ". => lookup_table")

        cleanCheckout = true
    }

    steps {
        script {
            name = "Run examples"
            id = "Run_examples"
            workingDir = "imod-python"
            scriptContent = """
                SET PATH=%%PATH%%;%system.teamcity.build.checkoutDir%\modflow6
                pixi run --environment default --frozen examples
            """.trimIndent()
            formatStderrAsError = true
            dockerImage = "%DockerContainer%:%DockerVersion%"
            dockerImagePlatform = ScriptBuildStep.ImagePlatform.Windows
            dockerRunParameters = """--cpus=8 --memory=32g"""
            dockerPull = false
        }
    }

    features {
        dockerSupport {
            loginToRegistry = on {
                dockerRegistryId = "PROJECT_EXT_134"
            }
        }
        xmlReport {
            reportType = XmlReport.XmlReportType.JUNIT
            rules = "imod-python/imod/tests/*report.xml"
        }
    }
})