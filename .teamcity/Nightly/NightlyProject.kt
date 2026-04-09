package Nightly

import Templates.*
import jetbrains.buildServer.configs.kotlin.AbsoluteId
import jetbrains.buildServer.configs.kotlin.BuildType
import jetbrains.buildServer.configs.kotlin.FailureAction
import jetbrains.buildServer.configs.kotlin.Project
import jetbrains.buildServer.configs.kotlin.buildFeatures.notifications
import jetbrains.buildServer.configs.kotlin.triggers.finishBuildTrigger

object NightlyProject : Project({
    name = "Nightly"

    buildType(NightlyLint)
    buildType(NightlyMyPy)
    buildType(NightlyUnitTests)
    buildType(NightlyExamples)
    buildType(NightlyTests)
    buildType(NightlyPipPython)
})

object NightlyLint : BuildType({
    name = "Lint"

    templates(LintTemplate)
})

object NightlyPipPython : BuildType({
    name = "PipPython"

    templates(PipPythonTemplate)
})

object NightlyMyPy : BuildType({
    name = "MyPy"

    templates(MyPyTemplate)
})

object NightlyUnitTests : BuildType({
    name = "UnitTests"

    templates(UnitTestsTemplate)

    dependencies {
        snapshot(NightlyLint) {
            onDependencyFailure = FailureAction.FAIL_TO_START
        }
        snapshot(NightlyMyPy) {
            onDependencyFailure = FailureAction.FAIL_TO_START
        }
        dependency(AbsoluteId("iMOD6_Modflow6buildWin64")) {
            snapshot {
                onDependencyFailure = FailureAction.FAIL_TO_START
            }

            artifacts {
                artifactRules = "+:MODFLOW6.zip!** => modflow6"
            }
        }
    }
})

object NightlyExamples : BuildType({
    name = "Examples"

    templates(ExamplesTemplate)

    dependencies {
        snapshot(NightlyLint) {
            onDependencyFailure = FailureAction.FAIL_TO_START
        }
        snapshot(NightlyMyPy) {
            onDependencyFailure = FailureAction.FAIL_TO_START
        }
        dependency(AbsoluteId("iMOD6_Modflow6buildWin64")) {
            snapshot {
                onDependencyFailure = FailureAction.FAIL_TO_START
            }

            artifacts {
                artifactRules = "+:MODFLOW6.zip!** => modflow6"
            }
        }
    }
})

object NightlyTests : BuildType({
    name = "Tests"

    allowExternalStatus = true
    type = Type.COMPOSITE

    vcs {
        root(AbsoluteId("iMOD6_IMODPython_ImodPython"))

        branchFilter = """
            +:*
            -:release_imod56
        """.trimIndent()
        showDependenciesChanges = true
    }

    triggers {
        finishBuildTrigger {
            buildType = "iMOD6_Modflow6buildWin64"
            successfulOnly = true
        }
    }

    features {
        notifications {
            notifierSettings = emailNotifier {
                email = """
                    joeri.vanengelen@deltares.nl
                    luitjan.slooten@deltares.nl
                    sunny.titus@deltares.nl
                """.trimIndent()
            }
            buildFailedToStart = true
            buildFailed = true
        }
    }

    dependencies {
        snapshot(NightlyPipPython) {
            onDependencyFailure = FailureAction.FAIL_TO_START
        }
        snapshot(NightlyExamples) {
            onDependencyFailure = FailureAction.FAIL_TO_START
        }
        dependency(NightlyUnitTests) {
            snapshot {
                onDependencyFailure = FailureAction.FAIL_TO_START
            }

            artifacts {
                artifactRules = "+:coverage.zip => ."
            }
        }
    }
})

