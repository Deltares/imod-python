package Templates

import jetbrains.buildServer.configs.kotlin.DslContext
import jetbrains.buildServer.configs.kotlin.Template
import jetbrains.buildServer.configs.kotlin.buildFeatures.XmlReport
import jetbrains.buildServer.configs.kotlin.buildFeatures.dockerSupport
import jetbrains.buildServer.configs.kotlin.buildFeatures.xmlReport
import jetbrains.buildServer.configs.kotlin.buildSteps.ScriptBuildStep
import jetbrains.buildServer.configs.kotlin.buildSteps.powerShell
import jetbrains.buildServer.configs.kotlin.buildSteps.script

object UnitTestsTemplate : Template({
    name = "UnitTestsTemplate"

    allowExternalStatus = true
    artifactRules = """
        imod-python\imod\tests\temp => test_output.zip
        imod-python\imod\tests\coverage => coverage.zip
    """.trimIndent()

    vcs {
        root(DslContext.settingsRoot, "+:. => imod-python")

        cleanCheckout = true
    }

    steps {
        script {
            name = "Run unittests"
            id = "Run_unittests"
            workingDir = "imod-python"
            scriptContent = """
                SET PATH=%%PATH%%;%system.teamcity.build.checkoutDir%\modflow6
                pixi run --environment default --frozen unittests
            """.trimIndent()
            formatStderrAsError = true
            dockerImage = "containers.deltares.nl/hydrology_product_line_imod/windows-pixi:v0.26.1"
            dockerImagePlatform = ScriptBuildStep.ImagePlatform.Windows
            dockerPull = true
        }
        powerShell {
            name = "Extract coverage statistics"
            id = "Extract_coverage_statistics"
            workingDir = "imod-python/imod/tests"
            scriptMode = script {
                content = """
                    ${'$'}REPORT = echo "coverage report" | pixi shell --environment default
                    
                    ${'$'}TOTALS = ${'$'}REPORT | Select-String -Pattern 'TOTAL' -CaseSensitive -SimpleMatch
                    ${'$'}STATISTICS = ${'$'}TOTALS -split "\s+"
                    ${'$'}TOTALLINES = ${'$'}STATISTICS[1]
                    ${'$'}MISSEDLINES = ${'$'}STATISTICS[2]
                    ${'$'}COVEREDLINES = ${'$'}TOTALLINES - ${'$'}MISSEDLINES
                    
                    Write-Host "##teamcity[buildStatisticValue key='CodeCoverageAbsLCovered' value='${'$'}COVEREDLINES']"
                    Write-Host "##teamcity[buildStatisticValue key='CodeCoverageAbsLTotal' value='${'$'}TOTALLINES']"
                """.trimIndent()
            }
            param("plugin.docker.imagePlatform", "windows")
            param("plugin.docker.imageId", "containers.deltares.nl/hydrology_product_line_imod/windows-pixi:v0.26.1")
            param("plugin.docker.pull.enabled", "true")
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