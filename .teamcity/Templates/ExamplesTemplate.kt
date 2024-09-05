package Templates

import jetbrains.buildServer.configs.kotlin.AbsoluteId
import jetbrains.buildServer.configs.kotlin.DslContext
import jetbrains.buildServer.configs.kotlin.Template
import jetbrains.buildServer.configs.kotlin.buildFeatures.XmlReport
import jetbrains.buildServer.configs.kotlin.buildFeatures.xmlReport
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
                set Path=%system.teamcity.build.checkoutDir%\modflow6;"d:\ProgramData\pixi"
                pixi run --environment default --frozen examples
            """.trimIndent()
            formatStderrAsError = true
        }
    }

    features {
        xmlReport {
            reportType = XmlReport.XmlReportType.JUNIT
            rules = "imod-python/imod/tests/*report.xml"
        }
    }

    requirements {
        equals("env.OS", "Windows_NT")
        doesNotExist("container.engine")
    }
})