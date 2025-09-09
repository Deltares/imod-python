package Weekly

import Templates.AcceptanceTestsTemplate
import jetbrains.buildServer.configs.kotlin.*
import jetbrains.buildServer.configs.kotlin.buildFeatures.notifications
import jetbrains.buildServer.configs.kotlin.triggers.ScheduleTrigger
import jetbrains.buildServer.configs.kotlin.triggers.schedule

object WeeklyProject : Project({
    name = "Weekly"

    buildType(AcceptanceTests)
    buildType(WeeklyTests)

})

object AcceptanceTests : BuildType({
    name = "AcceptanceTests"

    templates(AcceptanceTestsTemplate)

    dependencies {
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

object  WeeklyTests : BuildType({
    name = "Tests"

    allowExternalStatus = true
    type = Type.COMPOSITE

    vcs {
        root(DslContext.settingsRoot)

        branchFilter = """
            +:*
            -:release_imod56
        """.trimIndent()
        showDependenciesChanges = true
    }


    triggers {
        schedule {
            schedulingPolicy = weekly {
                dayOfWeek = ScheduleTrigger.DAY.Sunday
                hour = 16
                minute = 0
            }
            branchFilter = "+:<default>"
            triggerBuild = always()
            withPendingChangesOnly = false
        }
    }

    failureConditions {
        errorMessage = true
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
        snapshot(AcceptanceTests) {
            onDependencyFailure = FailureAction.FAIL_TO_START
        }
    }
})