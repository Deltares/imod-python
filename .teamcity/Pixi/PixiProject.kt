package Pixi

import jetbrains.buildServer.configs.kotlin.AbsoluteId
import jetbrains.buildServer.configs.kotlin.BuildType
import jetbrains.buildServer.configs.kotlin.Project
import jetbrains.buildServer.configs.kotlin.buildSteps.powerShell
import jetbrains.buildServer.configs.kotlin.triggers.schedule

object PixiProject : Project({
    name = "Pixi"

    buildType(UpdateDependencies)
})

object UpdateDependencies : BuildType({
    name = "Update Dependencies"

    params {
        param("GH_USER", "deltares-service-account")
        text("env.GH_TOKEN", "%github_deltares-service-account_access_token%")
    }

    vcs {
        root(AbsoluteId("iMOD6_IMODPython_ImodPython"), "+:. => imod-python")

        cleanCheckout = true
    }

    steps {
        powerShell {
            name = "Update dependencies"
            id = "Update_dependencies"
            workingDir = "imod-python"
            scriptMode = script {
                content = """
                    echo "Create update branch"
                    git remote set-url origin https://%GH_USER%:%env.GH_TOKEN%@github.com/Deltares/imod-python.git
                    git checkout -b pixi_update_%build.counter%
                    
                    echo "Update dependencies" 
                    pixi run -e pixi-update update
                    
                    echo "Add any changes" 
                    git add pixi.lock
                    
                    if (git status -suno) 
                    {
                      git commit -m "Update pixi.lock"
                      git push -u origin pixi_update_%build.counter%

                      echo Teamcity automatically updated the dependencies defined the pixi.toml file. Please verify that all tests succeed before merging. > body.md
                      echo. >> body.md
                      type diff.md >> body.md
                      pixi run --environment default gh pr create --title "[TEAMCITY] Update project dependencies" --body-file body.md --reviewer JoerivanEngelen,luitjansl
                      echo "Changes pushed and PR created"
                    }
                    else
                    {
                      echo "No changes found"
                    }
                """.trimIndent()
            }
            noProfile = false
        }
    }

    triggers {
        schedule {
            schedulingPolicy = weekly {
                hour = 15
            }
            branchFilter = "+:<default>"
            triggerBuild = always()
            withPendingChangesOnly = false
        }
    }

    failureConditions {
        errorMessage = true
    }
})