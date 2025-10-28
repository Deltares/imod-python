package Docker

import jetbrains.buildServer.configs.kotlin.BuildType
import jetbrains.buildServer.configs.kotlin.Project
import jetbrains.buildServer.configs.kotlin.PublishMode

object DockerProject  : Project({
    name = "Docker dev project"

    buildType(DeployDockerDevImage)
})

object DeployDockerDevImage : BuildType({
    name = "Deploy docker dev image"

    enablePersonalBuilds = false
    type = Type.DEPLOYMENT
    maxRunningBuilds = 1
    publishArtifacts = PublishMode.SUCCESSFUL

})