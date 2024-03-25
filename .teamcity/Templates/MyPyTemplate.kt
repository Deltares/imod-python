package Templates

import jetbrains.buildServer.configs.kotlin.DslContext
import jetbrains.buildServer.configs.kotlin.Template
import jetbrains.buildServer.configs.kotlin.buildFeatures.XmlReport
import jetbrains.buildServer.configs.kotlin.buildFeatures.xmlReport
import jetbrains.buildServer.configs.kotlin.buildSteps.script
import jetbrains.buildServer.configs.kotlin.failureConditions.BuildFailureOnMetric
import jetbrains.buildServer.configs.kotlin.failureConditions.failOnMetricChange

object MyPyTemplate : Template({
    name = "MyPyTemplate"

    detectHangingBuilds = false

    vcs {
        root(DslContext.settingsRoot, "+:. => imod-python")

        cleanCheckout = true
    }

    steps {
        script {
            name = "MyPy analysis"
            id = "MyPy_analysis"
            workingDir = "imod-python"
            scriptContent = """
                    pixi run --environment default --frozen mypy_report
                    pixi run --environment default --frozen mypy
                """.trimIndent()
            formatStderrAsError = true
        }
    }

    failureConditions {
        nonZeroExitCode = false
        testFailure = false
        failOnMetricChange {
            metric = BuildFailureOnMetric.MetricType.TEST_FAILED_COUNT
            threshold = 0
            units = BuildFailureOnMetric.MetricUnit.DEFAULT_UNIT
            comparison = BuildFailureOnMetric.MetricComparison.MORE
            compareTo = value()
        }
    }

    features {
        xmlReport {
            reportType = XmlReport.XmlReportType.JUNIT
            rules = "imod-python/*.xml"
        }
    }

    requirements {
        equals("env.OS", "Windows_NT")
    }
})